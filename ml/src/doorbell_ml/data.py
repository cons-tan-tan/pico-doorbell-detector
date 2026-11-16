import csv
import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .audio import (
    Frontend,
    augment_event,
    crop_background,
    crop_event,
    feature_sequence,
    features,
    load_audio,
    mix_at_snr,
    random_level,
    shift_event,
)

LABELS = ("background", "doorbell_type1", "doorbell_type2")
SPLITS = ("train", "val", "test")
FIELDS = ("path", "label", "split", "group", "source", "license", "kind")


@dataclass(frozen=True)
class Recording:
    path: Path
    label: str
    split: str
    group: str
    source: str = ""
    license: str = ""
    kind: str = "recorded"


@dataclass(frozen=True)
class HardNegative:
    recording: Recording
    start_sample: int | None = None


def read_manifests(paths: list[Path]) -> list[Recording]:
    records = []
    groups = {}
    contents = {}
    seen_paths = set()
    for manifest in paths:
        with manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(FIELDS[:4]).issubset(reader.fieldnames or []):
                raise ValueError(f"{manifest}: required columns are {FIELDS[:4]}")
            for row in reader:
                path = (manifest.parent / row["path"]).resolve()
                record = Recording(
                    path=path,
                    label=row["label"],
                    split=row["split"],
                    group=row["group"],
                    source=row.get("source", ""),
                    license=row.get("license", ""),
                    kind=row.get("kind") or "recorded",
                )
                if (
                    record.label not in LABELS
                    or record.split not in SPLITS
                    or not record.group
                ):
                    raise ValueError(
                        f"{manifest}: invalid label, split, or group for {path}"
                    )
                if record.kind not in ("recorded", "synthetic"):
                    raise ValueError(f"{path}: kind must be recorded or synthetic")
                if record.kind == "synthetic" and (
                    record.label != "background" or record.split != "train"
                ):
                    raise ValueError(
                        "Synthetic instruments are training negatives, not held-out recordings"
                    )
                if not path.is_file():
                    raise ValueError(f"Audio file not found: {path}")
                if path in seen_paths:
                    raise ValueError(f"Duplicate path in manifests: {path}")
                seen_paths.add(path)
                previous = groups.setdefault(record.group, record.split)
                if previous != record.split:
                    raise ValueError(
                        f"Group {record.group!r} appears in both {previous} and {record.split}"
                    )
                digest = file_digest(path)
                if row.get("sha256") and digest != row["sha256"]:
                    raise ValueError(f"Audio checksum mismatch: {path}")
                previous = contents.setdefault(digest, (record.split, record.label))
                if previous != (record.split, record.label):
                    raise ValueError(
                        f"Duplicate audio has conflicting splits or labels: {path}"
                    )
                records.append(record)
    if not records:
        raise ValueError("No recordings in manifests")
    return records


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        fields = FIELDS + (("sha256",) if any("sha256" in row for row in rows) else ())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict | list) -> None:
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(content)


def inventory(records: list[Recording]) -> dict:
    return {
        split: {
            label: {
                "files": sum(r.split == split and r.label == label for r in records),
                "groups": len(
                    {r.group for r in records if r.split == split and r.label == label}
                ),
            }
            for label in LABELS
        }
        for split in SPLITS
    }


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def choose_positive(records: list[Recording], rng: np.random.Generator) -> Recording:
    groups = {}
    for record in records:
        groups.setdefault(record.group, []).append(record)
    choices = list(groups.values())
    group = choices[int(rng.integers(len(choices)))]
    return group[int(rng.integers(len(group)))]


def check_independent(records, metadata: dict, split: str, extra_sources=()) -> None:
    used_splits = {"train", "val"} if split == "test" else {"train"}
    used = [r for r in metadata["sources"] if r["split"] in used_splits]
    used.extend(extra_sources)
    used_groups = {r["group"] for r in used}
    used_hashes = {r["sha256"] for r in used}
    for record in records:
        if record.group in used_groups or file_digest(record.path) in used_hashes:
            raise ValueError(
                f"Evaluation recording overlaps model development data: {record.path}"
            )


def load_hard_negatives(
    path: Path | None, records: list[Recording]
) -> list[HardNegative]:
    if path is None:
        return []
    report = json.loads(path.read_text())
    if report.get("split") != "train":
        raise ValueError("Hard negatives must come from the training split")
    available = {
        str(r.path): r
        for r in records
        if r.split == "train" and r.label == "background" and r.kind == "recorded"
    }
    selected = []
    for row in report["recordings"]:
        record = available.get(row["path"])
        if (
            record is None
            or record.group != row["group"]
            or file_digest(record.path) != row["sha256"]
        ):
            raise ValueError(
                "Hard-negative sources must match recorded training backgrounds"
            )
        start = row.get("start_sample")
        if start is not None and (
            type(start) is not int
            or report.get("sample_rate") != 8000
            or report.get("window_samples") != 16000
            or not 0 <= start <= max(0, len(load_audio(record.path)) - 16000)
        ):
            raise ValueError("Invalid hard-negative window")
        selected.append(HardNegative(record, start))
    if not selected or len({item.recording for item in selected}) != len(selected):
        raise ValueError(
            "Hard-negative list must be nonempty and contain unique sources"
        )
    return selected


def make_examples(
    records: list[Recording],
    split: str,
    per_class: int,
    seed: int,
    config: Frontend,
    synthetic_fraction: float = 0.25,
    background_multiplier: int = 1,
    hard_negatives: list[HardNegative] | None = None,
    hard_negative_fraction: float = 0.5,
    time_shift_seconds: float = 0.5,
    *,
    soundscape_fraction: float = 0.0,
    min_snr_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        per_class < 1
        or background_multiplier < 1
        or not 0 <= synthetic_fraction <= 1
        or not 0 <= hard_negative_fraction <= 1
        or not 0 <= soundscape_fraction <= 1
        or not np.isfinite(min_snr_db)
        or not -10 <= min_snr_db <= 25
        or not 0 <= time_shift_seconds < config.window_samples / config.sample_rate
    ):
        raise ValueError(
            "Example counts must be positive and sampling fractions must be in [0, 1]"
        )
    pool = [r for r in records if r.split == split]
    missing = set(LABELS) - {r.label for r in pool}
    if missing:
        raise ValueError(f"{split} lacks recordings for: {', '.join(sorted(missing))}")
    rng = np.random.default_rng(seed)
    backgrounds = [r for r in pool if r.label == "background" and r.kind == "recorded"]
    synthetic = [r for r in pool if r.kind == "synthetic"]
    if not backgrounds:
        raise ValueError(f"{split} needs at least one recorded background")
    if hard_negatives and (
        split != "train"
        or not {item.recording for item in hard_negatives}.issubset(backgrounds)
    ):
        raise ValueError("Hard negatives must be recorded backgrounds from train")

    def context_before(source, start):
        if config.compression != "pcen":
            return None
        context = source[max(0, start - config.window_samples) : start]
        return np.pad(context, (config.window_samples - len(context), 0))

    def background():
        if synthetic and rng.random() < synthetic_fraction:
            choices = synthetic
        elif hard_negatives and rng.random() < hard_negative_fraction:
            item = hard_negatives[int(rng.integers(len(hard_negatives)))]
            source = load_audio(item.recording.path, config.sample_rate)
            if item.start_sample is not None:
                samples = source[
                    item.start_sample : item.start_sample + config.window_samples
                ]
                return np.pad(
                    samples, (0, config.window_samples - len(samples))
                ), context_before(source, item.start_sample)
            choices = [item.recording]
        else:
            choices = backgrounds
        record = choices[int(rng.integers(len(choices)))]
        source = load_audio(record.path, config.sample_rate)
        if config.compression == "pcen" and len(source) >= config.window_samples:
            start = int(rng.integers(len(source) - config.window_samples + 1))
            return source[start : start + config.window_samples].copy(), context_before(
                source, start
            )
        return crop_background(source, config.window_samples, rng), context_before(
            source, 0
        )

    with ExitStack() as stack:
        scaped = {label: [] for label in LABELS[1:]}
        if soundscape_fraction:
            from .soundscapes import SoundscapeCorpus, stream_windows

            corpus = stack.enter_context(SoundscapeCorpus(pool, config))
            for label in LABELS[1:]:
                target = round(per_class * soundscape_fraction)
                choices = [record for record in pool if record.label == label]
                while len(scaped[label]) < target:
                    record = choose_positive(choices, rng)
                    noise = backgrounds[int(rng.integers(len(backgrounds)))]
                    snr = float(rng.uniform(min_snr_db, 25))
                    audio, events = corpus.render(
                        noise,
                        [(record, float(rng.uniform(2, 3)), snr)],
                        rng,
                        augment=split == "train",
                    )
                    windows, truth, times = stream_windows(audio, events, config)
                    indices = np.flatnonzero(truth == LABELS.index(label))
                    if not len(indices):
                        raise ValueError("No identifiable chime windows in soundscape")
                    for index in rng.choice(
                        indices,
                        size=min(4, len(indices), target - len(scaped[label])),
                        replace=False,
                    ):
                        start = (
                            round(times[index] * config.sample_rate)
                            - config.window_samples
                        )
                        scaped[label].append(
                            (windows[index], context_before(audio, start))
                        )

    waveforms, labels, contexts = [], [], []
    for label_id, label in enumerate(LABELS):
        choices = [r for r in pool if r.label == label]
        count = per_class * (background_multiplier if label_id == 0 else 1)
        for _ in range(count):
            noise, context = background()
            quiet = False
            if label_id == 0:
                samples = noise
                if split == "train" and rng.random() < 0.1:
                    quiet = True
                    level = (
                        0 if rng.random() < 0.25 else 10 ** (rng.uniform(-80, -45) / 20)
                    )
                    samples = rng.normal(0, level, config.window_samples)
                    if context is not None:
                        context = np.zeros(config.window_samples, dtype=np.float32)
            else:
                record = choose_positive(choices, rng)
                if scaped[label]:
                    samples, context = scaped[label].pop()
                else:
                    samples = load_audio(record.path, config.sample_rate)
                    if split == "train":
                        samples = augment_event(samples, rng, config)
                    samples = crop_event(samples, config.window_samples, rng)
                    if split == "train" and time_shift_seconds:
                        samples = shift_event(
                            samples, round(time_shift_seconds * config.sample_rate), rng
                        )
                    samples = mix_at_snr(
                        samples,
                        noise,
                        float(rng.uniform(min_snr_db, 25)),
                    )
            if not quiet:
                before = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
                if split == "train":
                    samples = random_level(samples, rng, low_db=-45, high_db=-6)
                else:
                    samples = random_level(samples, rng)
                if context is not None:
                    after = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
                    context = np.clip(context * after / max(before, 1e-10), -1, 1)
            waveforms.append(samples)
            contexts.append(context)
            labels.append(label_id)
    order = rng.permutation(len(labels))
    pcm = np.asarray(waveforms, dtype=np.float32)[order]
    if config.compression == "pcen":
        x = np.stack(
            [
                feature_sequence(
                    np.concatenate((contexts[index], waveforms[index])), config
                )[-config.shape[0] :]
                for index in order
            ]
        )
    else:
        x = np.stack([features(samples, config) for samples in pcm])
    y = np.asarray(labels, dtype=np.int64)[order]
    return x, y, pcm


def provenance(records: list[Recording]) -> list[dict]:
    result = []
    for record in records:
        row = asdict(record)
        row["path"] = str(record.path)
        row["sha256"] = file_digest(record.path)
        result.append(row)
    return result

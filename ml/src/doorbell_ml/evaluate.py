import json

import numpy as np

from .audio import Frontend, feature_sequence, features, load_audio
from .data import LABELS, check_independent, file_digest, read_manifests, write_json
from .metrics import alarm_times, classification_metrics
from .model import Int8Model


def window_starts(length: int, window: int, hop: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, hop))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def score_audio(
    runtime,
    audio: np.ndarray,
    config: Frontend,
    hop_seconds: float,
    *,
    streaming: bool = False,
) -> dict:
    if hop_seconds <= 0:
        raise ValueError("Hop must be positive")
    hop = max(1, round(hop_seconds * config.sample_rate))
    starts = (
        list(range(0, len(audio) - config.window_samples + 1, hop))
        if streaming
        else window_starts(len(audio), config.window_samples, hop)
    )
    windows = []
    sequences = {}
    for start in starts:
        samples = audio[start : start + config.window_samples]
        samples = np.pad(samples, (0, config.window_samples - len(samples)))
        if config.compression == "pcen" and len(audio) >= config.window_samples:
            phase = start % config.hop_samples
            if phase not in sequences:
                sequences[phase] = feature_sequence(audio[phase:], config)
            frame = start // config.hop_samples
            windows.append(sequences[phase][frame : frame + config.shape[0]])
        else:
            windows.append(features(samples, config))
    times = (
        np.minimum(np.asarray(starts) + config.window_samples, len(audio))
        / config.sample_rate
    )
    return {
        "scores": runtime.predict(np.stack(windows))
        if windows
        else np.empty((0, len(LABELS))),
        "times": times,
        "seconds": len(audio) / config.sample_rate,
    }


def mine(args) -> dict:
    if args.count < 1:
        raise ValueError("Hard-negative count must be positive")
    records = [
        r
        for r in read_manifests(args.manifest)
        if r.split == "train" and r.label == "background" and r.kind == "recorded"
    ]
    if not records:
        raise ValueError("No recorded training backgrounds to mine")
    metadata = json.loads((args.model / "metadata.json").read_text())
    config = Frontend(**metadata["frontend"])
    model_path = args.model / "model.int8.tflite"
    runtime = Int8Model(model_path.read_bytes(), metadata)
    ranked = []
    for record in records:
        scored = score_audio(
            runtime,
            load_audio(record.path, config.sample_rate),
            config,
            args.hop_seconds,
        )
        peak = int(scored["scores"][:, 1:].max(axis=1).argmax())
        ranked.append(
            {
                "path": str(record.path),
                "group": record.group,
                "sha256": file_digest(record.path),
                "max_chime_score": float(scored["scores"][:, 1:].max()),
                "start_sample": window_starts(
                    len(load_audio(record.path, config.sample_rate)),
                    config.window_samples,
                    max(1, round(args.hop_seconds * config.sample_rate)),
                )[peak],
            }
        )
    ranked.sort(key=lambda row: (-row["max_chime_score"], row["path"]))
    report = {
        "split": "train",
        "model_sha256": file_digest(model_path),
        "hop_seconds": args.hop_seconds,
        "sample_rate": config.sample_rate,
        "window_samples": config.window_samples,
        "candidates": len(ranked),
        "recordings": ranked[: args.count],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"Selected {len(report['recordings'])} of {len(ranked)} training backgrounds")
    return report


def evaluate(args) -> dict:
    if (
        not 0 < args.threshold <= 1
        or args.hop_seconds <= 0
        or args.cooldown_seconds < 0
        or args.consecutive < 1
    ):
        raise ValueError("Invalid threshold, hop, or cooldown")
    metadata = json.loads((args.model / "metadata.json").read_text())
    config = Frontend(**metadata["frontend"])
    records = [r for r in read_manifests(args.manifest) if r.split == args.split]
    if not records:
        raise ValueError(f"No recordings in {args.split}")
    check_independent(records, metadata, args.split)
    runtime = Int8Model((args.model / "model.int8.tflite").read_bytes(), metadata)
    truth, decisions, details = [], [], []
    background_seconds, false_alarms = 0.0, 0
    hop = max(1, round(args.hop_seconds * config.sample_rate))
    for record in records:
        audio = load_audio(record.path, config.sample_rate)
        scored = score_audio(runtime, audio, config, args.hop_seconds)
        alarms = alarm_times(
            scored["scores"],
            scored["times"],
            args.threshold,
            args.cooldown_seconds,
            args.consecutive,
        )
        decision = np.zeros(len(LABELS), dtype=np.float32)
        if alarms:
            selected = max(alarms, key=lambda alarm: alarm["score"])
            decision[LABELS.index(selected["label"])] = 1
        else:
            decision[0] = 1
        truth.append(LABELS.index(record.label))
        decisions.append(decision)
        duration = len(audio) / config.sample_rate
        if record.label == "background":
            background_seconds += duration
            false_alarms += len(alarms)
        details.append(
            {
                "path": str(record.path),
                "label": record.label,
                "seconds": duration,
                "alarms": alarms,
            }
        )
    report = {
        "split": args.split,
        "clip_metrics": classification_metrics(
            np.asarray(truth), np.stack(decisions), args.threshold
        ),
        "background_hours": background_seconds / 3600,
        "background_false_alarms": false_alarms,
        "background_false_alarms_per_hour": false_alarms * 3600 / background_seconds
        if background_seconds
        else None,
        "hop_seconds": hop / config.sample_rate,
        "cooldown_seconds": args.cooldown_seconds,
        "consecutive": args.consecutive,
        "recordings": details,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "recordings"},
            indent=2,
        )
    )
    return report

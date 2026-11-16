import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import audio as audio_module
from .audio import Frontend, load_audio, streaming_features
from .data import file_digest, write_json
from .metrics import compact
from .model import float_predictions
from .policy_eval import (
    make_full_episode,
    normal_regressions,
    summarize_policy,
    validate_policy,
)
from .replay import run_decisions


def joint_selection_key(candidates, references):
    losses = [0, 0, 0]
    for candidate, reference in zip(candidates, references, strict=True):
        normal, base = candidate["suites"]["normal"], reference["suites"]["normal"]
        for index, loss in enumerate(normal_regressions(normal, base)):
            losses[index] += loss
    normal = [candidate["suites"]["normal"] for candidate in candidates]
    suites = [row for candidate in candidates for row in candidate["suites"].values()]
    return (
        any(losses),
        *losses,
        sum(row["background_false_alarms"] for row in normal),
        -sum(row["any_chime_detected"] for row in normal),
        -sum(row["any_chime_detected"] for row in suites),
        -sum(row["first_correct_kind"] for row in suites),
        sum(row["wrong_type_alarms"] for row in suites),
        sum(row["median_presence_delay_seconds"] or 100 for row in normal),
        -candidates[0]["policy"]["low"],
    )


def select_policy(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.reports or len(args.reports) != len(args.references):
        raise ValueError("Supply a matching reference for each validation corpus")
    reports = [json.loads(path.read_text()) for path in args.reports]
    references = [json.loads(path.read_text()) for path in args.references]
    decisions = [
        json.loads((path.parent / "decision.json").read_text()) for path in args.reports
    ]
    identity = ("model_sha256", "replay_sha256", "decision_sha256")
    first = decisions[0]
    conditions = (
        "seed",
        "episodes_per_condition",
        "snr_db",
        "suites",
        "timing",
        "positive_sources",
        "background_sources",
        "warmup_seconds",
        "post_event_seconds",
        "insertion_seconds",
        "speed_factor_range",
        "rms_dbfs",
        "snr_definition",
        "reverb",
    )
    sources, seen_backgrounds, pools = {}, set(), []
    for report, reference, decision in zip(reports, references, decisions, strict=True):
        if (
            report["split"] != "val"
            or reference["split"] != "val"
            or report["protocol"]["version"] != 7
            or reference["protocol"]["version"] != 7
            or any(
                report["protocol"][key] != reference["protocol"][key]
                for key in conditions
            )
            or any(decision[key] != first[key] for key in identity)
            or report["protocol"]["model_sha256"] != first["model_sha256"]
            or report["protocol"]["replay_sha256"] != first["replay_sha256"]
        ):
            raise ValueError(
                "Joint selection requires one model and matched validation conditions"
            )
        validate_policy(decision, *(first[key] for key in identity))
        for row in report["protocol"]["background_sources"]:
            if row["split"] != "val" or row["sha256"] in seen_backgrounds:
                raise ValueError(
                    "Validation corpora must contain distinct validation backgrounds"
                )
            seen_backgrounds.add(row["sha256"])
        for row in decision["sources"]:
            sources[(row["path"], row["sha256"])] = row
        pool = {}
        for candidate in report["validation_candidates"]:
            validate_policy(
                first | {"policy": candidate["policy"]},
                *(first[key] for key in identity),
            )
            key = json.dumps(candidate["policy"], sort_keys=True)
            if key in pool:
                raise ValueError("Duplicate validation policy")
            pool[key] = candidate
        pools.append(pool)
    common = sorted(set.intersection(*(set(pool) for pool in pools)))
    if not common:
        raise ValueError("Validation corpora have no shared policy candidates")
    baseline = [report["selected"] for report in references]
    key = min(
        common,
        key=lambda key: joint_selection_key([pool[key] for pool in pools], baseline),
    )
    selected = [pool[key] for pool in pools]
    rank = joint_selection_key(selected, baseline)
    selection = {
        "normal_non_regression": not rank[0],
        "rank": list(rank),
        "priority": "no normal regression in each corpus or chime kind; then total background alarms, presence and first-kind accuracy; latency breaks accuracy ties",
        "reports": [
            {"path": str(path.resolve()), "sha256": file_digest(path)}
            for path in args.reports
        ],
        "reference_reports": [
            {"path": str(path.resolve()), "sha256": file_digest(path)}
            for path in args.references
        ],
    }
    output = {
        "split": "val",
        "policy": selected[0]["policy"],
        "selection": selection,
        "corpora": [
            {"report": str(path.resolve()), "selected": candidate}
            for path, candidate in zip(args.reports, selected, strict=True)
        ],
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "report.json", output)
    write_json(
        args.output / "decision.json",
        first
        | {
            "policy": output["policy"],
            "sources": list(sources.values()),
            "selection": selection,
        },
    )
    print(json.dumps({"policy": output["policy"], "selection": selection}, indent=2))
    return output


def cache_validation(args):
    report = json.loads(args.report.read_text())
    protocol = report["protocol"]
    if report["split"] != "val" or protocol["version"] not in (6, 7):
        raise ValueError(
            "Checkpoint selection requires a notification validation report"
        )
    with gzip.open(args.trace, "rt") as handle:
        trace = json.load(handle)
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    if trace["fingerprint"] != fingerprint:
        raise ValueError("Validation trace does not match the report")
    sources = protocol["positive_sources"] + protocol["background_sources"]
    if any(file_digest(Path(row["path"])) != row["sha256"] for row in sources):
        raise ValueError("Validation audio changed since the report was generated")
    if any(row["split"] != "val" for row in protocol["background_sources"]):
        raise ValueError("Checkpoint selection must not use test backgrounds")
    config = Frontend(compression=getattr(args, "compression", "log"))
    rows = trace["backgrounds"] + trace["episodes"]
    args.output.mkdir(parents=True, exist_ok=False)
    count = sum(len(row["times"]) for row in rows)
    x = np.lib.format.open_memmap(
        args.output / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, *config.shape),
    )
    position = 0
    for index, row in enumerate(rows):
        if "path" in row:
            audio = load_audio(Path(row["path"]))
        else:
            audio, onset, offset = make_full_episode(
                SimpleNamespace(path=Path(row["positive"])),
                SimpleNamespace(path=Path(row["background"])),
                row["snr_db"],
                row["augmentation_seed"],
                row["suite"],
                config,
            )
            if (onset, offset) != (row["onset"], row["offset"]):
                raise ValueError("Validation episode regeneration changed")
        inputs = streaming_features(audio, config)
        times = np.arange(1, len(inputs) + 1) * 0.256
        if len(inputs) != len(row["times"]) or not np.allclose(
            times, row["times"], atol=1e-6
        ):
            raise ValueError("Validation feature windows do not match replay timing")
        x[position : position + len(inputs)] = inputs
        row["feature_start"] = position
        row["id"] = index
        position += len(inputs)
    x.flush()
    write_json(
        args.output / "validation.json",
        {
            "split": "val",
            "frontend": config.to_dict(),
            "sources": sources,
            "report_sha256": file_digest(args.report),
            "trace_sha256": file_digest(args.trace),
            "features_sha256": file_digest(args.output / "features.npy"),
            "frontend_source_sha256": file_digest(Path(audio_module.__file__)),
            "background_count": len(trace["backgrounds"]),
            "rows": rows,
            "policy": report["selected"]["policy"],
            "engine": "ADC simulation and NumPy frontend; float model on int8 output grid; Rust decisions",
        },
    )
    print(f"Cached {count} windows from {len(rows)} validation clips", flush=True)


def notification_key(summary):
    suites = list(summary["suites"].values())
    return (
        suites[0]["background_false_alarms"],
        -min(row["presence_recall"] for row in suites),
        -sum(row["any_chime_detected"] for row in suites),
        sum(row["wrong_type_alarms"] for row in suites),
        -sum(row["first_correct_kind"] for row in suites),
        sum(row["median_presence_delay_seconds"] or 100 for row in suites),
    )


class NotificationValidator:
    def __init__(self, directory, binary, config, sources):
        self.metadata = json.loads((directory / "validation.json").read_text())
        if self.metadata.get("frontend_source_sha256") != file_digest(
            Path(audio_module.__file__)
        ):
            raise ValueError("Rebuild notification validation after frontend changes")
        if (
            self.metadata["split"] != "val"
            or Frontend(**self.metadata["frontend"]) != config
        ):
            raise ValueError("Notification validation split or frontend does not match")
        used = [row for row in sources if row["split"] == "train"]
        groups, hashes = {row["group"] for row in used}, {row["sha256"] for row in used}
        for row in self.metadata["sources"]:
            if file_digest(Path(row["path"])) != row["sha256"]:
                raise ValueError("Notification validation source audio changed")
            if row["label"] == "background" and row["split"] != "val":
                raise ValueError(
                    "Notification validation requires validation backgrounds"
                )
            if row["label"] == "background" and (
                row["group"] in groups or row["sha256"] in hashes
            ):
                raise ValueError(
                    "Notification validation overlaps training backgrounds"
                )
        if file_digest(directory / "features.npy") != self.metadata["features_sha256"]:
            raise ValueError("Notification validation features changed")
        if binary is None or not binary.is_file():
            raise ValueError(
                "Notification selection needs a Rust decision replay binary"
            )
        self.x = np.load(directory / "features.npy", mmap_mode="r")
        self.binary = binary
        self.rows = self.metadata["rows"]
        self.background_count = self.metadata["background_count"]

    def evaluate(self, model, inputs=None):
        probabilities = float_predictions(model, self.x if inputs is None else inputs)
        if (
            probabilities.shape != (len(self.x), 3)
            or not np.isfinite(probabilities).all()
        ):
            raise ValueError(
                "Notification predictions do not match the fixed validation windows"
            )
        scores = np.clip(np.rint(probabilities * 256) - 128, -128, 127).astype(np.int8)
        rows = []
        for row in self.rows:
            start = row["feature_start"]
            values = iter(scores[start : start + len(row["times"])])
            trace = []
            for line in row["trace"]:
                if line.startswith("prediction,"):
                    kind, time, *_ = line.split(",")
                    line = f"{kind},{time}," + ",".join(
                        str(int(value)) for value in next(values)
                    )
                trace.append(line)
            rows.append(row | {"trace": trace})
        policy = self.metadata["policy"]
        decisions = run_decisions(self.binary, rows, policy)
        summary = summarize_policy(
            rows[: self.background_count],
            rows[self.background_count :],
            "fusion",
            policy["low"],
            policy,
            decisions,
        )
        return {
            "policy": policy,
            "suites": {
                name: compact(value) for name, value in summary["suites"].items()
            },
        }

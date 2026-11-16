import json
from dataclasses import replace

import numpy as np

from .audio import Frontend, load_audio
from .benchmark import make_episode
from .data import (
    LABELS,
    check_independent,
    file_digest,
    provenance,
    read_manifests,
    write_json,
)
from .metrics import compact, metrics
from .replay import score_clip


def select_threshold(rules, candidates, minimum_snr):
    for row in candidates:
        row["lost_rule_detections"] = sum(
            old["detected"] and not new["detected"]
            for old, new in zip(
                rules["episode_details"], row["episode_details"], strict=True
            )
        )
        row["lost_rule_detections_at_selection_snr"] = sum(
            old["detected"] and not new["detected"] and old["snr_db"] >= minimum_snr
            for old, new in zip(
                rules["episode_details"], row["episode_details"], strict=True
            )
        )
    return min(
        candidates,
        key=lambda row: (
            row["lost_rule_detections_at_selection_snr"],
            row["background_false_alarms"],
            row["wrong_type_alarms"],
            -row["detected"],
            -row["threshold"],
        ),
    )


def compare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.split == "test" and args.policy is None:
        raise ValueError("Test comparison requires a policy selected on val")
    if args.split == "val" and args.policy is not None:
        raise ValueError("Select the policy on val before applying it to test")
    if (
        args.episodes_per_condition < 1
        or not args.snr_db
        or not np.isfinite(args.snr_db).all()
        or not np.isfinite(args.minimum_snr_db)
        or max(args.snr_db) < args.minimum_snr_db
        or not args.thresholds
        or any(not 0 < t <= 1 for t in args.thresholds)
    ):
        raise ValueError("Invalid comparison settings")
    metadata = json.loads((args.model / "metadata.json").read_text())
    config = Frontend(**metadata["frontend"])
    if replace(config, compression="log") != Frontend():
        raise ValueError("The Rust rules require the firmware's 8000 Hz frontend")
    model = args.model / "model.int8.tflite"
    model_sha, replay_sha = file_digest(model), file_digest(args.replay_binary)
    timing = {
        "hop_seconds": 0.256,
        "confirmation_seconds": 1.0,
        "cooldown_seconds": 3.0,
    }
    frozen = json.loads(args.policy.read_text()) if args.policy else None
    if args.policy is not None and (
        not isinstance(frozen, dict)
        or frozen.get("calibration_split") != "val"
        or frozen.get("model_sha256") != model_sha
        or frozen.get("replay_sha256") != replay_sha
        or frozen.get("timing") != timing
        or frozen.get("minimum_selection_snr_db") != args.minimum_snr_db
        or not isinstance(frozen.get("threshold"), (int, float))
        or not 0 < frozen["threshold"] <= 1
        or not isinstance(frozen.get("sources"), list)
        or not frozen["sources"]
    ):
        raise ValueError("Policy must match the model, Rust rules, and timing")
    records = read_manifests(args.manifest)
    backgrounds = sorted(
        (r for r in records if r.split == args.split and r.label == "background"),
        key=lambda r: str(r.path),
    )
    positives = sorted(
        (r for r in records if r.split == "train" and r.label != "background"),
        key=lambda r: str(r.path),
    )
    if not backgrounds or {r.label for r in positives} != set(LABELS[1:]):
        raise ValueError("Need held-out backgrounds and both known chimes")
    check_independent(
        backgrounds, metadata, args.split, frozen["sources"] if frozen else ()
    )
    negatives, episodes = [], []
    for row in backgrounds:
        negatives.append(
            score_clip(
                args.replay_binary, load_audio(row.path), config, metadata, model_sha
            )
            | {"path": str(row.path)}
        )
    rng = np.random.default_rng(args.seed)
    for label in LABELS[1:]:
        choices = [r for r in positives if r.label == label]
        for snr in args.snr_db:
            for _ in range(args.episodes_per_condition):
                positive = choices[int(rng.integers(len(choices)))]
                background = backgrounds[int(rng.integers(len(backgrounds)))]
                audio, onset, offset = make_episode(
                    positive, background, snr, rng, config
                )
                episodes.append(
                    score_clip(args.replay_binary, audio, config, metadata, model_sha)
                    | {
                        "label": label,
                        "snr_db": snr,
                        "onset": onset,
                        "offset": offset,
                        "positive": str(positive.path),
                        "background": str(background.path),
                    }
                )
    rules = metrics(negatives, episodes, "rules", None, timing)
    thresholds = [frozen["threshold"]] if frozen else sorted(set(args.thresholds))
    hybrids = [metrics(negatives, episodes, "hybrid", t, timing) for t in thresholds]
    selected = select_threshold(rules, hybrids, args.minimum_snr_db)
    ml_thresholds = sorted(set([0.95] + thresholds))
    ml = [metrics(negatives, episodes, "ml", t, timing) for t in ml_thresholds]
    protocol = {
        "seed": args.seed,
        "snr_db": args.snr_db,
        "episodes_per_condition": args.episodes_per_condition,
        "minimum_selection_snr_db": args.minimum_snr_db,
        "warmup_seconds": 2.0,
        "episode_seconds": 5,
        "event_window_seconds": 2,
        "insertion_seconds": [0.75, 1.75],
        "rms_dbfs": [-38, -15],
        "augmentation": "speed_0.96_to_1.04_lowpass_echo",
        "frontend": config.to_dict(),
        "verification": "same_type_score; latest_prediction_or_up_to_1s_after_candidate",
        "latency": "audio_timeline_only; excludes_compute_and_network_time",
        "positive_sources_independent": False,
        "background_sources": provenance(backgrounds),
        "positive_sources": provenance(positives),
        "model_sha256": model_sha,
        "replay_sha256": replay_sha,
        "engine": "Rust firmware frontend and rules, C++ TFLite Micro on host",
        "adc_simulation": "normalized_PCM_to_12bit; no microphone or room transfer function",
        "timing": timing,
    }
    report = {
        "split": args.split,
        "protocol": protocol,
        "rules": rules,
        "ml": ml,
        "hybrid": hybrids,
        "selected_threshold": selected["threshold"],
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "report.json", report)
    if not frozen:
        write_json(
            args.output / "decision.json",
            {
                "calibration_split": "val",
                "threshold": selected["threshold"],
                "model_sha256": model_sha,
                "replay_sha256": replay_sha,
                "timing": timing,
                "selection": "minimize_lost_rule_detections_then_false_alarms_wrong_types",
                "minimum_selection_snr_db": args.minimum_snr_db,
                "sources": provenance(backgrounds) + provenance(positives),
            },
        )
    print(
        json.dumps(
            {
                "rules": compact(rules),
                "ml": [compact(r) for r in ml],
                "hybrid": [compact(r) for r in hybrids],
                "selected_threshold": selected["threshold"],
            },
            indent=2,
        )
    )
    return report

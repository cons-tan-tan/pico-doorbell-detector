import hashlib
import json

import numpy as np

from .audio import (
    Frontend,
    augment_event,
    crop_background,
    crop_event,
    load_audio,
    mix_at_snr,
    random_level,
)
from .data import (
    LABELS,
    check_independent,
    file_digest,
    provenance,
    read_manifests,
    write_json,
)
from .evaluate import score_audio
from .metrics import alarm_times
from .model import Int8Model


def make_episode(positive, background, snr, rng, config):
    length = 5 * config.sample_rate
    noise = load_audio(background.path, config.sample_rate)
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    noise = crop_background(noise, length, rng)
    event = augment_event(load_audio(positive.path, config.sample_rate), rng, config)
    event = crop_event(event, config.window_samples, rng)
    onset = int(rng.uniform(0.75, 1.75) * config.sample_rate)
    foreground = np.zeros(length, dtype=np.float32)
    foreground[onset : onset + len(event)] = event
    mixture = random_level(mix_at_snr(foreground, noise, snr), rng)
    return (
        mixture,
        onset / config.sample_rate,
        (onset + len(event)) / config.sample_rate,
    )


def summarize(backgrounds, episodes, policy, window_seconds, include_details=False):
    counts = {}
    false_alarms, flagged, background_seconds = 0, 0, 0.0
    background_details, episode_details = [], []
    for row in backgrounds:
        alarms = alarm_times(
            row["scores"],
            row["times"],
            policy["threshold"],
            policy["cooldown_seconds"],
            policy["consecutive"],
        )
        false_alarms += len(alarms)
        flagged += bool(alarms)
        background_seconds += row["seconds"]
        if include_details:
            background_details.append(
                {"path": row["path"], "seconds": row["seconds"], "alarms": alarms}
            )
    for row in episodes:
        alarms = alarm_times(
            row["scores"],
            row["times"],
            policy["threshold"],
            policy["cooldown_seconds"],
            policy["consecutive"],
        )
        matches = [
            alarm
            for alarm in alarms
            if alarm["label"] == row["label"]
            and row["onset"] < alarm["seconds"] <= row["offset"] + window_seconds
        ]
        key = (row["label"], row["snr_db"])
        count = counts.setdefault(
            key, {"episodes": 0, "detected": 0, "wrong_type_alarms": 0, "delays": []}
        )
        count["episodes"] += 1
        count["detected"] += bool(matches)
        count["wrong_type_alarms"] += sum(a["label"] != row["label"] for a in alarms)
        if matches:
            count["delays"].append(matches[0]["seconds"] - row["onset"])
        if include_details:
            episode_details.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("scores", "times")
                }
                | {"detected": bool(matches), "alarms": alarms}
            )
    conditions = [
        {
            "label": label,
            "snr_db": snr,
            "episodes": count["episodes"],
            "detected": count["detected"],
            "recall": count["detected"] / count["episodes"],
            "wrong_type_alarms": count["wrong_type_alarms"],
            "mean_detection_seconds_after_insert": float(np.mean(count["delays"]))
            if count["delays"]
            else None,
        }
        for (label, snr), count in sorted(counts.items())
    ]
    result = {
        "policy": policy,
        "independent_background": {
            "clips": len(backgrounds),
            "seconds": background_seconds,
            "clips_with_false_alarms": flagged,
            "false_alarms": false_alarms,
            "false_alarms_per_hour": false_alarms * 3600 / background_seconds,
        },
        "known_chime_robustness": {
            "positive_sources_independent": False,
            "conditions": conditions,
            "episodes": len(episodes),
            "detected": sum(c["detected"] for c in conditions),
            "recall": sum(c["detected"] for c in conditions) / len(episodes),
            "wrong_type_alarms": sum(c["wrong_type_alarms"] for c in conditions),
        },
    }
    if include_details:
        result["background_recordings"] = background_details
        result["episodes"] = episode_details
    return result


def select_policy(candidates, minimum_recall, minimum_snr):
    scored = []
    for candidate in candidates:
        conditions = [
            c
            for c in candidate["known_chime_robustness"]["conditions"]
            if c["snr_db"] >= minimum_snr
        ]
        if not conditions:
            raise ValueError("No benchmark conditions meet the selection SNR")
        recall = min(c["recall"] for c in conditions)
        wrong = sum(c["wrong_type_alarms"] for c in conditions)
        false_alarms = candidate["independent_background"]["false_alarms"]
        meets_target = recall >= minimum_recall
        policy = candidate["policy"]
        key = (
            (0, false_alarms, wrong, -recall)
            if meets_target
            else (1, -recall, false_alarms, wrong)
        ) + (policy["consecutive"], -policy["threshold"])
        scored.append((key, candidate, recall))
    _, selected, recall = min(scored, key=lambda item: item[0])
    return selected, {
        "minimum_recall": minimum_recall,
        "minimum_snr_db": minimum_snr,
        "worst_condition_recall": recall,
        "target_met": recall >= minimum_recall,
        "priority": "meet_recall_target_then_minimize_background_alarms_and_wrong_types",
    }


def benchmark(args) -> dict:
    if args.calibrate and args.split != "val":
        raise ValueError("Decision calibration is allowed only on val, never test")
    if args.calibrate and args.policy:
        raise ValueError("Choose calibration or an existing policy")
    if (
        args.episodes_per_condition < 1
        or args.hop_seconds <= 0
        or args.cooldown_seconds < 0
        or not 0 < args.minimum_recall <= 1
        or not args.snr_db
        or not np.isfinite(args.snr_db).all()
    ):
        raise ValueError("Invalid benchmark settings")
    if args.output.exists():
        raise FileExistsError(f"Benchmark output already exists: {args.output}")
    metadata = json.loads((args.model / "metadata.json").read_text())
    config = Frontend(**metadata["frontend"])
    model_path = args.model / "model.int8.tflite"
    model_sha = file_digest(model_path)
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
        raise ValueError(
            "Benchmark needs held-out backgrounds and both training chimes"
        )
    policy_file = json.loads(args.policy.read_text()) if args.policy else None
    if args.policy and (
        not isinstance(policy_file, dict)
        or policy_file.get("model_sha256") != model_sha
        or policy_file.get("calibration_split") != "val"
        or not isinstance(policy_file.get("policy"), dict)
        or policy_file["policy"].get("hop_seconds") != args.hop_seconds
        or policy_file["policy"].get("cooldown_seconds") != args.cooldown_seconds
    ):
        raise ValueError(
            "Policy must match this model and the validation timing settings"
        )
    check_independent(
        backgrounds,
        metadata,
        args.split,
        policy_file["sources"] if policy_file and args.split == "test" else (),
    )
    background_sources = provenance(backgrounds)
    positive_sources = provenance(positives)
    protocol = {
        "version": 2,
        "window_schedule": "complete_windows_on_prediction_grid",
        "seed": args.seed,
        "episodes_per_condition": args.episodes_per_condition,
        "snr_db": args.snr_db,
        "hop_seconds": args.hop_seconds,
        "episode_seconds": 5,
        "insertion_seconds": [0.75, 1.75],
        "rms_dbfs": [-38, -15],
        "augmentation": "speed_0.96_to_1.04_lowpass_echo",
        "frontend": config.to_dict(),
        "background_sources": background_sources,
        "positive_sources": positive_sources,
    }
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    runtime = Int8Model(model_path.read_bytes(), metadata)
    background_scores, episodes = [], []
    for record in backgrounds:
        scored = score_audio(
            runtime,
            load_audio(record.path, config.sample_rate),
            config,
            args.hop_seconds,
            streaming=True,
        )
        background_scores.append(scored | {"path": str(record.path)})
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
                scored = score_audio(
                    runtime, audio, config, args.hop_seconds, streaming=True
                )
                episodes.append(
                    scored
                    | {
                        "positive": str(positive.path),
                        "background": str(background.path),
                        "label": label,
                        "snr_db": snr,
                        "onset": onset,
                        "offset": offset,
                    }
                )
    default_policy = {
        "threshold": 0.8,
        "consecutive": 1,
        "hop_seconds": args.hop_seconds,
        "cooldown_seconds": args.cooldown_seconds,
    }
    window_seconds = config.window_samples / config.sample_rate
    default = summarize(background_scores, episodes, default_policy, window_seconds)
    selected_policy = policy_file["policy"] if policy_file else default_policy
    candidates, selection = [], None
    if args.calibrate:
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98):
            for consecutive in (1, 2, 3):
                policy = default_policy | {
                    "threshold": threshold,
                    "consecutive": consecutive,
                }
                candidates.append(
                    summarize(background_scores, episodes, policy, window_seconds)
                )
        selected, selection = select_policy(
            candidates, args.minimum_recall, args.minimum_snr_db
        )
        selected_policy = selected["policy"]
    selected = summarize(
        background_scores, episodes, selected_policy, window_seconds, True
    )
    report = {
        "split": args.split,
        "model_sha256": model_sha,
        "protocol_sha256": protocol_sha,
        "protocol": protocol,
        "independent_positive_recordings": 0,
        "default": default,
        "selected": selected,
        "selection": selection,
        "calibration_candidates": candidates,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    if args.calibrate:
        write_json(
            args.output / "decision.json",
            {
                "model_sha256": model_sha,
                "calibration_split": "val",
                "protocol_sha256": protocol_sha,
                "positive_sources_independent": False,
                "policy": selected_policy,
                "selection": selection,
                "sources": background_sources + positive_sources,
            },
        )
    print(
        json.dumps(
            {
                "split": args.split,
                "protocol_sha256": protocol_sha,
                "selection": selection,
                "default": default,
                "selected": {
                    k: v
                    for k, v in selected.items()
                    if k not in ("background_recordings", "episodes")
                },
            },
            indent=2,
        )
    )
    return report

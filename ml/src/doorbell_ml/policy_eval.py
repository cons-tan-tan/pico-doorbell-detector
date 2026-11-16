import gzip
import hashlib
import json
from dataclasses import replace

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .audio import Frontend, crop_background, load_audio, mix_at_snr, rms
from .data import (
    LABELS,
    check_independent,
    file_digest,
    provenance,
    read_manifests,
    write_json,
)
from .metrics import compact, metrics
from .replay import run_decisions, score_clip

SUITES = ("normal", "pitch", "reverb", "combined")
TIMING = {"hop_seconds": 0.256, "confirmation_seconds": 1.0, "cooldown_seconds": 10.0}


def make_full_episode(positive, background, snr, seed, suite, config):
    rng = np.random.default_rng(seed)
    event = load_audio(positive.path, config.sample_rate).copy()
    event -= event.mean()
    speed = int(rng.integers(960, 1041))
    cutoff = float(rng.uniform(1800, 3600))
    delay = int(rng.uniform(0.025, 0.12) * config.sample_rate)
    echo_gain = float(rng.uniform(0.05, 0.25))
    onset = int(rng.uniform(0.75, 1.75) * config.sample_rate)
    level_db = float(rng.uniform(-38, -15))
    if suite in ("pitch", "combined"):
        event = resample_poly(event, 1000, speed)
    if suite in ("reverb", "combined"):
        event = sosfilt(butter(2, cutoff, fs=config.sample_rate, output="sos"), event)
        event = np.pad(event, (0, delay)) + echo_gain * np.pad(event, (delay, 0))
    offset = onset + len(event)
    length = offset + 2 * config.sample_rate
    source = load_audio(background.path, config.sample_rate)
    if len(source) < length:
        source = np.tile(source, int(np.ceil(length / len(source))))
    noise = crop_background(source, length, rng)
    foreground = np.zeros(length, dtype=np.float32)
    foreground[onset:offset] = event
    mixture = mix_at_snr(foreground, noise, snr)
    mixture = np.clip(
        mixture * (10 ** (level_db / 20) / max(rms(mixture), 1e-10)), -1, 1
    ).astype(np.float32)
    return mixture, onset / config.sample_rate, offset / config.sample_rate


def summarize_policy(
    backgrounds, episodes, method, threshold, policy=None, decisions=None
):
    provider = None if decisions is None else lambda row: decisions[row["id"]]
    return {
        "policy": policy,
        "suites": {
            suite: metrics(
                backgrounds,
                [r for r in episodes if r["suite"] == suite],
                method,
                threshold,
                TIMING,
                event_provider=provider,
            )
            for suite in SUITES
        },
    }


def normal_regressions(normal, base):
    per_kind = []
    for report in (normal, base):
        counts = {}
        for row in report.get("conditions", []):
            values = counts.setdefault(row["label"], [0, 0])
            values[0] += row["presence_detected"]
            values[1] += row["detected"]
        per_kind.append(counts)
    if per_kind[0].keys() != per_kind[1].keys():
        raise ValueError("Reference and candidate must report the same chime kinds")
    kind_losses = [
        sum(
            max(0, per_kind[1][label][index] - per_kind[0][label][index])
            for label in per_kind[0]
        )
        for index in (0, 1)
    ]
    return (
        max(0, normal["background_false_alarms"] - base["background_false_alarms"]),
        max(kind_losses[0], base["any_chime_detected"] - normal["any_chime_detected"]),
        max(
            kind_losses[1],
            base.get("first_correct_kind", base["detected"])
            - normal.get("first_correct_kind", normal["detected"]),
        ),
    )


def selection_key(candidate, baseline):
    normal, base = candidate["suites"]["normal"], baseline["suites"]["normal"]
    losses = normal_regressions(normal, base)
    return (
        any(losses),
        *losses,
        normal["background_false_alarms"],
        -normal["any_chime_detected"],
        -sum(r["any_chime_detected"] for r in candidate["suites"].values()),
        -normal.get("first_correct_kind", normal["detected"]),
        sum(r["wrong_type_alarms"] for r in candidate["suites"].values()),
        normal.get(
            "median_presence_delay_seconds", normal["median_audio_delay_seconds"]
        )
        or float("inf"),
        -candidate["policy"]["low"],
    )


def validate_policy(frozen, model_sha, replay_sha, decision_sha):
    if (
        not isinstance(frozen, dict)
        or frozen.get("calibration_split") != "val"
        or frozen.get("protocol_version") not in (4, 5, 6, 7)
        or frozen.get("model_sha256") != model_sha
        or frozen.get("replay_sha256") != replay_sha
        or frozen.get("decision_sha256") != decision_sha
        or not isinstance(replay_sha, str)
        or not isinstance(decision_sha, str)
        or not isinstance(frozen.get("sources"), list)
        or not frozen["sources"]
        or not isinstance(frozen.get("policy"), dict)
    ):
        raise ValueError(
            "Policy must match the validation model, replay, and decision binaries"
        )
    policy = frozen["policy"]
    keys = {"low", "high", "confirmation_frames", "use_rules"}
    if frozen["protocol_version"] >= 5:
        keys.add("cooldown_seconds")
        seconds = policy.get("cooldown_seconds")
        if (
            type(seconds) not in (int, float)
            or not np.isfinite(seconds)
            or not 0 <= seconds <= (2**32 - 1) / 1000
        ):
            raise ValueError("Invalid notification cooldown")
    if frozen["protocol_version"] >= 6:
        keys.add("rule_hold_ms")
        hold = policy.get("rule_hold_ms")
        if type(hold) is not int or not 0 <= hold <= 2000:
            raise ValueError("Pattern hold must fit inside the inference window")
    if frozen["protocol_version"] >= 7:
        keys.add("minimum_support")
        support = policy.get("minimum_support")
        if (
            type(support) is not int
            or type(policy.get("confirmation_frames")) is not int
            or not 2 <= support <= policy["confirmation_frames"]
        ):
            raise ValueError("Invalid same-kind confirmation count")
    if (
        set(policy) != keys
        or any(
            not isinstance(policy[key], (int, float)) or isinstance(policy[key], bool)
            for key in ("low", "high")
        )
        or not 0 < policy["low"] <= policy["high"] <= 1
        or not isinstance(policy["confirmation_frames"], int)
        or policy["confirmation_frames"] not in (2, 3, 4)
        or not isinstance(policy["use_rules"], bool)
    ):
        raise ValueError("Invalid frozen decision policy")


def evaluate_policy(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.split == "test") != (args.policy is not None):
        raise ValueError("Calibrate on val; test requires a frozen validation policy")
    fresh = getattr(args, "fresh_test_backgrounds", False)
    if fresh and args.split != "test":
        raise ValueError("Fresh-background declaration applies only to test evaluation")
    if (
        args.episodes_per_condition < 1
        or not args.snr_db
        or not np.isfinite(args.snr_db).all()
    ):
        raise ValueError("Invalid episode count or SNR")
    metadata = json.loads((args.model / "metadata.json").read_text())
    config = Frontend(**metadata["frontend"])
    if replace(config, compression="log") != Frontend():
        raise ValueError("Unsupported firmware frontend")
    model_sha = file_digest(args.model / "model.int8.tflite")
    replay_sha, decision_sha = (
        file_digest(args.replay_binary),
        file_digest(args.decision_binary),
    )
    frozen = json.loads(args.policy.read_text()) if args.policy else None
    if args.policy:
        validate_policy(frozen, model_sha, replay_sha, decision_sha)
        if frozen["protocol_version"] != 7:
            raise ValueError("Recalibrate on val for same-kind temporal confirmation")
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
    protocol = {
        "version": 7,
        "timing": TIMING,
        "seed": args.seed,
        "episodes_per_condition": args.episodes_per_condition,
        "snr_db": args.snr_db,
        "suites": SUITES,
        "positive_sources": provenance(positives),
        "background_sources": provenance(backgrounds),
        "positive_sources_independent": False,
        "event_crop": "none; full recording retained",
        "warmup_seconds": 2,
        "post_event_seconds": 2,
        "insertion_seconds": [0.75, 1.75],
        "speed_factor_range": [0.96, 1.04],
        "rms_dbfs": [-38, -15],
        "snr_definition": "active foreground samples above 5% peak versus full background RMS",
        "reverb": "lowpass_1800_3600Hz_echo_25_120ms_gain_0.05_0.25",
        "model_sha256": model_sha,
        "replay_sha256": replay_sha,
        "metadata_sha256": file_digest(args.model / "metadata.json"),
        "latency": "audio timeline; excludes compute and notification transport",
        "test_backgrounds_previously_inspected": args.split == "test" and not fresh,
    }
    reference_path = getattr(args, "reference_report", None)
    reference = None
    if reference_path:
        reference_report = json.loads(reference_path.read_text())
        if (
            args.split != "val"
            or reference_report["split"] != "val"
            or any(
                reference_report["protocol"][key] != protocol[key]
                for key in (
                    "seed",
                    "episodes_per_condition",
                    "snr_db",
                    "positive_sources",
                    "background_sources",
                )
            )
        ):
            raise ValueError(
                "Reference detector must use the same validation audio and conditions"
            )
        reference = reference_report["selected"]
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    if args.cache and args.cache.exists():
        with gzip.open(args.cache, "rt") as handle:
            cached = json.load(handle)
        if cached["fingerprint"] != fingerprint:
            raise ValueError("Replay cache does not match sources, protocol, or binary")
        negatives, episodes = cached["backgrounds"], cached["episodes"]
        for row in negatives + episodes:
            row["scores"], row["times"] = (
                np.asarray(row["scores"]),
                np.asarray(row["times"]),
            )
    else:
        negatives, episodes = [], []
        for recording in backgrounds:
            negatives.append(
                score_clip(
                    args.replay_binary,
                    load_audio(recording.path),
                    config,
                    metadata,
                    model_sha,
                )
                | {"path": str(recording.path)}
            )
        rng = np.random.default_rng(args.seed)
        for label in LABELS[1:]:
            choices = [r for r in positives if r.label == label]
            for snr in args.snr_db:
                for _ in range(args.episodes_per_condition):
                    positive = choices[int(rng.integers(len(choices)))]
                    background = backgrounds[int(rng.integers(len(backgrounds)))]
                    seed = int(rng.integers(2**32))
                    for suite in SUITES:
                        audio, onset, offset = make_full_episode(
                            positive, background, snr, seed, suite, config
                        )
                        episodes.append(
                            score_clip(
                                args.replay_binary, audio, config, metadata, model_sha
                            )
                            | {
                                "label": label,
                                "snr_db": snr,
                                "suite": suite,
                                "onset": onset,
                                "offset": offset,
                                "positive": str(positive.path),
                                "background": str(background.path),
                                "augmentation_seed": seed,
                            }
                        )
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(args.cache, "xt") as handle:
                json.dump(
                    {
                        "fingerprint": fingerprint,
                        "backgrounds": negatives,
                        "episodes": episodes,
                    },
                    handle,
                    default=lambda value: (
                        value.tolist() if isinstance(value, np.ndarray) else value
                    ),
                )
    rows = negatives + episodes
    for index, row in enumerate(rows):
        row["id"] = index
    baselines = {
        name: summarize_policy(negatives, episodes, method, threshold)
        for name, method, threshold in (
            ("rules", "rules", None),
            ("tolerant_rules", "tolerant_rules", None),
            ("ml_95", "ml", 0.95),
            ("ml_60", "ml", 0.6),
            ("cascade_60", "hybrid", 0.6),
            ("tolerant_cascade_60", "tolerant_cascade", 0.6),
        )
    }
    policies = (
        [frozen["policy"]]
        if frozen
        else [
            {
                "low": low,
                "high": high,
                "confirmation_frames": frames,
                "use_rules": True,
                "cooldown_seconds": TIMING["cooldown_seconds"],
                "rule_hold_ms": rule_hold,
                "minimum_support": support,
            }
            for low in (0.6, 0.8, 0.9)
            for high in (0.95, 0.98, 0.99)
            for frames in (2, 3, 4)
            for support in range(2, frames + 1)
            for rule_hold in (768, 1280)
        ]
    )
    candidates = []
    for policy in policies:
        decisions = run_decisions(args.decision_binary, rows, policy)
        candidates.append(
            summarize_policy(
                negatives, episodes, "fusion", policy["low"], policy, decisions
            )
        )
    selected = (
        candidates[0]
        if frozen
        else min(
            candidates,
            key=lambda candidate: selection_key(
                candidate, reference or baselines["ml_95"]
            ),
        )
    )
    control_policy = selected["policy"] | {"use_rules": False}
    control = summarize_policy(
        negatives,
        episodes,
        "temporal_ml",
        0.95,
        control_policy,
        run_decisions(args.decision_binary, rows, control_policy),
    )
    selection = {
        "normal_non_regression": not selection_key(
            selected, reference or baselines["ml_95"]
        )[0],
        "reference_report_sha256": file_digest(reference_path)
        if reference_path
        else None,
        "priority": "no extra reference background alarms, normal presence misses or wrong first kinds; then stress presence and kind accuracy; latency breaks accuracy ties",
    }
    report = {
        "split": args.split,
        "protocol": protocol,
        "baselines": baselines,
        "selected": selected,
        "temporal_control": control,
        "selection": selection,
        "validation_candidates": [
            {
                "policy": r["policy"],
                "suites": {s: compact(v) for s, v in r["suites"].items()},
            }
            for r in candidates
        ]
        if not frozen
        else [],
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "report.json", report)
    if not frozen:
        write_json(
            args.output / "decision.json",
            {
                "protocol_version": 7,
                "calibration_split": "val",
                "model_sha256": model_sha,
                "replay_sha256": replay_sha,
                "decision_sha256": decision_sha,
                "policy": selected["policy"],
                "sources": provenance(backgrounds) + provenance(positives),
                "selection": selection,
            },
        )
    print(
        json.dumps(
            {
                "policy": selected["policy"],
                "selection": selection,
                "selected": {s: compact(v) for s, v in selected["suites"].items()},
            },
            indent=2,
        )
    )
    return report

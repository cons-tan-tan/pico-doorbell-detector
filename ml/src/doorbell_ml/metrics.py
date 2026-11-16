import numpy as np

from .data import LABELS


def classification_metrics(
    y: np.ndarray, probabilities: np.ndarray, threshold: float = 0.8
) -> dict:
    predicted = probabilities.argmax(axis=1)
    predicted[probabilities.max(axis=1) < threshold] = 0
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    per_class = {}
    for index, label in enumerate(LABELS):
        support = int(confusion[index].sum())
        selected = int(confusion[:, index].sum())
        correct = int(confusion[index, index])
        per_class[label] = {
            "support": support,
            "precision": correct / selected if selected else None,
            "recall": correct / support if support else None,
        }
    negatives = y == 0
    return {
        "examples": len(y),
        "threshold": threshold,
        "accuracy": float(np.mean(predicted == y)),
        "confusion_matrix": confusion.tolist(),
        "labels": list(LABELS),
        "per_class": per_class,
        "background_false_positive_fraction": float(np.mean(predicted[negatives] != 0))
        if negatives.any()
        else None,
    }


def alarm_times(
    probabilities: np.ndarray,
    times: np.ndarray,
    threshold: float,
    cooldown: float,
    consecutive: int = 1,
) -> list[dict]:
    if not 0 < threshold <= 1 or cooldown < 0 or consecutive < 1:
        raise ValueError("Invalid threshold, cooldown, or consecutive count")
    alarms = []
    last = -float("inf")
    previous, count = 0, 0
    for probabilities_at_time, time in zip(probabilities, times, strict=True):
        label = int(probabilities_at_time.argmax())
        confidence = float(probabilities_at_time[label])
        qualified = label if label and confidence >= threshold else 0
        count = (
            count + 1 if qualified and qualified == previous else int(bool(qualified))
        )
        previous = qualified
        if count >= consecutive and time - last >= cooldown:
            alarms.append(
                {"seconds": float(time), "label": LABELS[label], "score": confidence}
            )
            last = time
    return alarms


def cooldown(events, seconds):
    accepted, last = [], -float("inf")
    for event in sorted(events, key=lambda item: item["seconds"]):
        if event["seconds"] - last >= seconds:
            accepted.append(event)
            last = event["seconds"]
    return accepted


def verify_candidates(
    candidates, scores, times, threshold, hop, wait, cooldown_seconds
):
    accepted = []
    for candidate in candidates:
        label = LABELS.index(candidate["label"])
        eligible = np.flatnonzero(
            (times >= candidate["seconds"] - hop - 1e-9)
            & (times <= candidate["seconds"] + wait + 1e-9)
            & (scores[:, label] >= threshold)
        )
        if len(eligible):
            index = eligible[0]
            accepted.append(
                candidate
                | {
                    "seconds": max(candidate["seconds"], float(times[index])),
                    "candidate_seconds": candidate["seconds"],
                    "score": float(scores[index, label]),
                }
            )
    return cooldown(accepted, cooldown_seconds)


def events_for(row, method, threshold, policy):
    if method in ("rules", "tolerant_rules"):
        return cooldown(
            row["patterns" if method == "tolerant_rules" else "candidates"],
            policy["cooldown_seconds"],
        )
    if method == "ml":
        return alarm_times(
            row["scores"], row["times"], threshold, policy["cooldown_seconds"]
        )
    return verify_candidates(
        row["patterns" if method == "tolerant_cascade" else "candidates"],
        row["scores"],
        row["times"],
        threshold,
        policy["hop_seconds"],
        policy["confirmation_seconds"],
        policy["cooldown_seconds"],
    )


def metrics(backgrounds, episodes, method, threshold, policy, event_provider=None):
    def events(row):
        return (
            event_provider(row)
            if event_provider
            else events_for(row, method, threshold, policy)
        )

    background_details, episode_details = [], []
    for row in backgrounds:
        background_details.append({"path": row["path"], "alarms": events(row)})
    for row in episodes:
        alarms = events(row)
        matching = [
            event
            for event in alarms
            if row["onset"] < event["seconds"] <= row["offset"] + 2.0
        ]
        correct = [event for event in matching if event["label"] == row["label"]]
        episode_details.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in (
                    "scores",
                    "times",
                    "candidates",
                    "patterns",
                    "trace",
                    "trace_version",
                    "decisions",
                )
            }
            | {
                "detected": bool(correct),
                "any_chime_detected": bool(matching),
                "first_type_correct": bool(matching)
                and matching[0]["label"] == row["label"],
                "presence_delay_seconds": matching[0]["seconds"] - row["onset"]
                if matching
                else None,
                "wrong_type_alarms": sum(a["label"] != row["label"] for a in alarms),
                "delay_seconds": correct[0]["seconds"] - row["onset"]
                if correct
                else None,
                "alarms": alarms,
            }
        )
    conditions = []
    for label, snr in sorted({(r["label"], r["snr_db"]) for r in episodes}):
        rows = [r for r in episode_details if (r["label"], r["snr_db"]) == (label, snr)]
        conditions.append(
            {
                "label": label,
                "snr_db": snr,
                "episodes": len(rows),
                "detected": sum(r["detected"] for r in rows),
                "recall": sum(r["detected"] for r in rows) / len(rows),
                "presence_detected": sum(r["any_chime_detected"] for r in rows),
                "presence_recall": sum(r["any_chime_detected"] for r in rows)
                / len(rows),
            }
        )
    seconds = sum(row["seconds"] for row in backgrounds)
    alarms = sum(len(row["alarms"]) for row in background_details)
    delays = [r["delay_seconds"] for r in episode_details if r["detected"]]
    presence_delays = [
        r["presence_delay_seconds"] for r in episode_details if r["any_chime_detected"]
    ]
    return {
        "method": method,
        "threshold": threshold if method != "rules" else None,
        "background_seconds": seconds,
        "background_false_alarms": alarms,
        "background_false_alarms_per_hour": alarms * 3600 / seconds,
        "episodes": len(episodes),
        "detected": sum(r["detected"] for r in episode_details),
        "recall": sum(r["detected"] for r in episode_details) / len(episodes),
        "any_chime_detected": sum(r["any_chime_detected"] for r in episode_details),
        "presence_recall": sum(r["any_chime_detected"] for r in episode_details)
        / len(episodes),
        "correct_kind_given_detection": sum(r["detected"] for r in episode_details)
        / sum(r["any_chime_detected"] for r in episode_details)
        if presence_delays
        else None,
        "first_correct_kind": sum(r["first_type_correct"] for r in episode_details),
        "first_type_accuracy": sum(r["first_type_correct"] for r in episode_details)
        / len(presence_delays)
        if presence_delays
        else None,
        "wrong_type_alarms": sum(r["wrong_type_alarms"] for r in episode_details),
        "median_audio_delay_seconds": float(np.median(delays)) if delays else None,
        "median_presence_delay_seconds": float(np.median(presence_delays))
        if presence_delays
        else None,
        "conditions": conditions,
        "background_recordings": background_details,
        "episode_details": episode_details,
    }


def compact(report):
    return {
        k: v
        for k, v in report.items()
        if k not in ("background_recordings", "episode_details")
    }


def notification_metrics(events, alarms, duration):
    assigned = set()
    details = []
    for index, event in enumerate(events):
        deadline = min(event["offset"] + 2, duration)
        if index + 1 < len(events):
            deadline = min(deadline, events[index + 1]["onset"])
        matched = [
            (i, alarm)
            for i, alarm in enumerate(alarms)
            if i not in assigned and event["onset"] < alarm["seconds"] <= deadline
        ]
        assigned.update(i for i, _ in matched)
        details.append(
            event
            | {
                "detected": bool(matched),
                "first_kind_correct": bool(matched)
                and matched[0][1]["label"] == event["label"],
                "wrong_kind_notifications": sum(
                    alarm["label"] != event["label"] for _, alarm in matched
                ),
                "duplicates": max(0, len(matched) - 1),
                "delay_seconds": matched[0][1]["seconds"] - event["onset"]
                if matched
                else None,
            }
        )
    delays = [row["delay_seconds"] for row in details if row["detected"]]
    return {
        "seconds": duration,
        "events": len(events),
        "detected": sum(row["detected"] for row in details),
        "first_kind_correct": sum(row["first_kind_correct"] for row in details),
        "wrong_kind_notifications": sum(
            row["wrong_kind_notifications"] for row in details
        ),
        "duplicate_notifications": sum(row["duplicates"] for row in details),
        "false_notifications": len(alarms) - len(assigned),
        "median_audio_delay_seconds": float(np.median(delays)) if delays else None,
        "p95_audio_delay_seconds": float(np.percentile(delays, 95)) if delays else None,
        "event_details": details,
        "false_notification_details": [
            alarm for i, alarm in enumerate(alarms) if i not in assigned
        ],
    }

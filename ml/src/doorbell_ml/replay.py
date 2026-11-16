import subprocess

import numpy as np

from .data import LABELS


def score_clip(binary, audio, config, metadata, model_sha):
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Replay requires nonempty finite mono PCM")
    warmup = config.window_samples / config.sample_rate
    pcm = np.pad(audio, (config.window_samples, 0)).astype("<f4")
    result = subprocess.run(
        [str(binary.resolve())], input=pcm.tobytes(), capture_output=True, check=True
    )
    lines = result.stdout.decode().splitlines()
    if not lines or lines[0] != f"model,{model_sha}":
        raise ValueError("Replay executable embeds a different model; rebuild it")
    if len(lines) < 2 or lines[1] != "hop_samples,2048":
        raise ValueError("Replay executable must use the 0.256-second prediction grid")
    version = (
        int(lines[2].split(",")[1])
        if len(lines) > 2 and lines[2].startswith("trace_version,")
        else 1
    )
    if version not in (1, 2, 3):
        raise ValueError("Unsupported replay trace version")
    candidates, patterns, scores, times, trace = [], [], [], [], []
    prediction_times = []
    previous = 0.0
    for line in lines[3 if version >= 2 else 2 :]:
        kind, seconds, *values = line.split(",")
        seconds = float(seconds)
        if (
            not np.isfinite(seconds)
            or seconds < previous
            or seconds > len(pcm) / config.sample_rate + 1e-6
        ):
            raise ValueError("Replay timestamps must follow the audio timeline")
        previous = seconds
        if kind == "prediction" and len(values) == len(LABELS):
            quantized = [int(value) for value in values]
            if any(not -128 <= value <= 127 for value in quantized):
                raise ValueError("Replay predictions must be int8 values")
            prediction_times.append(seconds)
        elif not (
            kind in ("candidate", "pattern")
            and len(values) == 1
            and values[0] in LABELS[1:]
            or kind == "tone"
            and not values
        ):
            raise ValueError(f"Invalid replay output: {line}")
        time = seconds - warmup
        if time <= 0:
            continue
        if kind == "candidate":
            candidates.append({"seconds": time, "label": values[0]})
        elif kind == "tone":
            trace.append(f"tone,{round(time * 1000)}")
        elif kind == "pattern":
            patterns.append({"seconds": time, "label": values[0]})
            trace.append(f"pattern,{round(time * 1000)},{values[0]}")
        else:
            trace.append(f"prediction,{round(time * 1000)},{','.join(values)}")
            scores.append(
                [
                    (value - metadata["output"]["zero_point"])
                    * metadata["output"]["scale"]
                    for value in quantized
                ]
            )
            times.append(time)
    expected = warmup + np.arange(len(audio) // 2048 + 1) * 2048 / config.sample_rate
    if len(prediction_times) != len(expected) or not np.allclose(
        prediction_times, expected, rtol=0, atol=1e-6
    ):
        raise ValueError("Replay predictions do not cover the complete prediction grid")
    return {
        "scores": np.asarray(scores).reshape(-1, len(LABELS)),
        "times": np.asarray(times),
        "seconds": len(audio) / config.sample_rate,
        "candidates": candidates,
        "patterns": patterns,
        "trace_version": version,
        "trace": trace,
    }


def run_decisions(binary, rows, policy):
    if not rows:
        return []
    lines = []
    for index, row in enumerate(rows):
        if row["trace_version"] != 3:
            raise ValueError(
                "Rebuild the replay executable to include tolerant patterns"
            )
        times = [
            int(line.split(",")[1])
            for line in row["trace"]
            if line.startswith("prediction,")
        ]
        expected = np.arange(1, round(row["seconds"] * 8000) // 2048 + 1) * 256
        if not np.array_equal(times, expected):
            raise ValueError(
                "Decision trace does not cover the complete prediction grid"
            )
        lines.append(f"reset,{index}")
        lines.extend(row["trace"])
    result = subprocess.run(
        [
            str(binary.resolve()),
            str(policy["low"]),
            str(policy["high"]),
            str(policy["confirmation_frames"]),
            str(policy["use_rules"]).lower(),
            *(
                [str(policy["cooldown_seconds"])]
                if "cooldown_seconds" in policy
                else []
            ),
            *([str(policy["rule_hold_ms"])] if "rule_hold_ms" in policy else []),
            *([str(policy["minimum_support"])] if "minimum_support" in policy else []),
        ],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    alarms = [[] for _ in rows]
    for line in result.stdout.splitlines():
        index, milliseconds, label = line.split(",")
        index, milliseconds = int(index), int(milliseconds)
        if (
            not 0 <= index < len(rows)
            or label not in LABELS[1:]
            or not 0 <= milliseconds <= round(rows[index]["seconds"] * 1000)
            or alarms[index]
            and milliseconds / 1000 < alarms[index][-1]["seconds"]
        ):
            raise ValueError("Decision replay returned an invalid notification")
        alarms[index].append({"seconds": milliseconds / 1000, "label": label})
    return alarms

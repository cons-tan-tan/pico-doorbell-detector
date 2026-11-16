import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audio import DEFAULT_FRONTEND, Frontend, coefficients
from .audio import features as audio_features
from .data import LABELS, file_digest, write_json
from .model import Int8Model, inspect_model


def verify_runtime(args):
    metadata = json.loads((args.model / "metadata.json").read_text())
    model_path = args.model / "model.int8.tflite"
    runtime = Int8Model(model_path.read_bytes(), metadata)
    with np.load(args.examples) as examples:
        x = examples["x"]
    if (
        x.ndim != 4
        or list(x.shape[1:]) != metadata["input"]["shape"][1:]
        or not len(x)
        or not np.isfinite(x).all()
    ):
        raise ValueError(
            "Runtime verification needs finite feature windows matching the model"
        )
    quantized = np.stack([runtime.quantize(sample) for sample in x])
    result = subprocess.run(
        [str(args.binary.resolve())],
        input=quantized.tobytes(),
        capture_output=True,
        check=True,
    )
    lines = result.stdout.decode().splitlines()
    if (
        len(lines) != len(x) + 2
        or lines[0] != f"model,{file_digest(model_path)}"
        or not lines[1].startswith("arena,")
    ):
        raise ValueError(
            "Inference replay binary does not match the verification model"
        )
    actual = np.asarray(
        [[int(value) for value in line.split(",")] for line in lines[2:]],
        dtype=np.int16,
    )
    expected = np.rint(
        runtime.predict(x) / metadata["output"]["scale"]
        + metadata["output"]["zero_point"]
    ).astype(np.int16)
    if actual.shape != expected.shape:
        raise ValueError("Invalid inference replay output shape")
    difference = np.abs(actual - expected)
    report = {
        "model_sha256": file_digest(model_path),
        "binary_sha256": file_digest(args.binary),
        "examples_sha256": file_digest(args.examples),
        "examples": len(x),
        "arena_bytes": int(lines[1].split(",")[1]),
        "max_integer_difference": int(difference.max()),
        "exact_output_fraction": float(np.mean(np.all(difference == 0, axis=1))),
        "argmax_agreement": float(
            np.mean(actual.argmax(axis=1) == expected.argmax(axis=1))
        ),
        "reference_runtime": "litert_builtin_reference",
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if report["max_integer_difference"] > 1:
        raise ValueError(
            "TFLite Micro outputs differ from reference by more than one quantization step"
        )
    return report


def export_firmware(model: Path, decision: Path, output: Path) -> None:
    metadata = json.loads((model / "metadata.json").read_text())
    selected = json.loads(decision.read_text())
    model_path = model / "model.int8.tflite"
    if (
        not isinstance(selected, dict)
        or selected.get("model_sha256") != file_digest(model_path)
        or selected.get("calibration_split") != "val"
    ):
        raise ValueError(
            "Decision settings must be selected on validation for this model"
        )
    config = Frontend(**metadata["frontend"])
    if replace(
        config, compression="log"
    ) != DEFAULT_FRONTEND or config.compression not in ("log", "pcen"):
        raise ValueError("Unsupported firmware audio frontend")
    if metadata.get("labels") != list(LABELS):
        raise ValueError("Firmware label order does not match")
    policy = selected.get("policy")
    if not isinstance(policy, dict):
        raise TypeError("Invalid firmware decision settings")
    if selected.get("protocol_version") in (4, 5, 6, 7):
        from .policy_eval import validate_policy

        validate_policy(
            selected,
            file_digest(model_path),
            selected.get("replay_sha256"),
            selected.get("decision_sha256"),
        )
        if selected.get("selection", {}).get("normal_non_regression") is not True:
            raise ValueError("Decision policy did not pass validation non-regression")
        policy = {
            "threshold": policy["high"],
            "consecutive": 1,
            "hop_seconds": 0.256,
            "cooldown_seconds": policy.get("cooldown_seconds", 3.0),
            "fusion": policy,
        }
    values = [
        policy.get(k)
        for k in ("threshold", "consecutive", "hop_seconds", "cooldown_seconds")
    ]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        raise ValueError("Invalid firmware decision settings")
    hop = policy["hop_seconds"] * DEFAULT_FRONTEND.sample_rate
    if (
        not np.isfinite(values).all()
        or not 0 < policy["threshold"] <= 1
        or not isinstance(policy["consecutive"], int)
        or policy["consecutive"] < 1
        or not 1 <= hop <= DEFAULT_FRONTEND.window_samples
        or not float(hop).is_integer()
        or not 0 <= policy["cooldown_seconds"] <= (2**32 - 1) / 1000
    ):
        raise ValueError("Invalid firmware decision settings")
    if int(hop) % DEFAULT_FRONTEND.hop_samples:
        raise ValueError("Firmware prediction hop must align with FFT frames")
    content = model_path.read_bytes()
    info = inspect_model(content)
    if info["input"]["shape"] != [1, *DEFAULT_FRONTEND.shape] or info["output"][
        "shape"
    ] != [1, 3]:
        raise ValueError("Unsupported firmware tensor shape")
    for key in ("input", "output", "operators"):
        if info[key] != metadata.get(key):
            raise ValueError(f"Model and metadata disagree: {key}")
    hann, mel = coefficients(DEFAULT_FRONTEND)
    with np.load(model / "frontend.npz") as frontend:
        if not np.array_equal(hann, frontend["hann"]) or not np.array_equal(
            mel, frontend["mel"]
        ):
            raise ValueError("Exported frontend coefficients do not match")
    runtime = Int8Model(content, info)
    reference_runtime = metadata.get("reference_runtime")
    if reference_runtime not in (None, "litert_builtin_reference"):
        raise ValueError("Unknown exported reference runtime")
    original_runtime = (
        runtime if reference_runtime else Int8Model(content, info, reference=False)
    )
    with np.load(model / "reference.npz") as reference:
        pcm = reference["pcm"]
        features = reference["features"]
        if pcm.shape != (16000,) or features.shape != DEFAULT_FRONTEND.shape:
            raise ValueError("Invalid reference shapes")
        if not np.isfinite(pcm).all() or not np.isfinite(features).all():
            raise ValueError("Non-finite reference data")
        if not np.allclose(features, audio_features(pcm, config), rtol=0, atol=1e-6):
            raise ValueError("Reference PCM and features do not match")
        quantized = runtime.quantize(features)
        if not np.array_equal(quantized, reference["input_int8"]):
            raise ValueError("Reference quantization does not match")
        original_probabilities = original_runtime.predict(features[np.newaxis])[0]
        if not np.allclose(
            original_probabilities, reference["output_probabilities"], atol=1e-7
        ):
            raise ValueError("Reference predictions do not match")
        probabilities = runtime.predict(features[np.newaxis])[0]
        output_int8 = np.rint(
            probabilities / info["output"]["scale"] + info["output"]["zero_point"]
        )
        files = {
            "model.tflite": content,
            "hann.f32": hann.astype("<f4").tobytes(),
            "mel.f32": mel.astype("<f4").tobytes(),
            "reference-pcm.f32": pcm.astype("<f4").tobytes(),
            "reference-features.f32": features.astype("<f4").tobytes(),
            "reference-input.i8": quantized.astype("i1").tobytes(),
            "reference-output.i8": output_int8.astype("i1").tobytes(),
        }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (output / name).write_bytes(data)
    write_json(
        output / "manifest.json",
        {
            "input": info["input"],
            "output": info["output"],
            "frontend": config.to_dict(),
            "labels": metadata["labels"],
            "policy": policy,
            "decision_sha256": file_digest(decision),
            "sha256": {
                name: hashlib.sha256(data).hexdigest() for name, data in files.items()
            },
            "pico_verified": False,
            "reference_runtime": "litert_builtin_reference",
            "source_reference_max_probability_difference": float(
                np.max(np.abs(probabilities - original_probabilities))
            ),
        },
    )

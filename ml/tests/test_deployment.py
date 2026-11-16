import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from doorbell_ml.audio import DEFAULT_FRONTEND, coefficients, features
from doorbell_ml.data import LABELS
from doorbell_ml.deployment import export_firmware


class DeploymentTests(unittest.TestCase):
    def test_legacy_accelerated_reference_is_reexported_with_integer_kernels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            content = b"model fixture"
            (model / "model.int8.tflite").write_bytes(content)
            info = {
                "input": {
                    "shape": [1, *DEFAULT_FRONTEND.shape],
                    "scale": 1 / 256,
                    "zero_point": -128,
                },
                "output": {"shape": [1, 3], "scale": 1 / 256, "zero_point": -128},
                "operators": [],
            }
            (model / "metadata.json").write_text(
                json.dumps(
                    info
                    | {
                        "frontend": DEFAULT_FRONTEND.to_dict(),
                        "labels": list(LABELS),
                    }
                )
            )
            hann, mel = coefficients(DEFAULT_FRONTEND)
            np.savez(model / "frontend.npz", hann=hann, mel=mel)
            pcm = np.zeros(16000, dtype=np.float32)
            x = features(pcm, DEFAULT_FRONTEND)
            quantized = np.full(x.shape, -128, dtype=np.int8)
            accelerated = (np.array([67, -74, -121]) + 128) / 256
            expected = np.array([60, -68, -120], dtype=np.int8)
            np.savez(
                model / "reference.npz",
                pcm=pcm,
                features=x,
                input_int8=quantized,
                output_probabilities=accelerated,
            )
            original = (model / "reference.npz").read_bytes()
            decision = root / "decision.json"
            decision.write_text(
                json.dumps(
                    {
                        "model_sha256": hashlib.sha256(content).hexdigest(),
                        "calibration_split": "val",
                        "policy": {
                            "threshold": 0.95,
                            "consecutive": 1,
                            "hop_seconds": 0.256,
                            "cooldown_seconds": 10,
                        },
                    }
                )
            )

            class Runtime:
                def __init__(self, content, info, reference=True):
                    self.probabilities = (
                        (expected.astype(float) + 128) / 256
                        if reference
                        else accelerated
                    )

                def quantize(self, sample):
                    return quantized

                def predict(self, samples):
                    return self.probabilities[np.newaxis]

            with (
                patch("doorbell_ml.deployment.inspect_model", return_value=info),
                patch("doorbell_ml.deployment.Int8Model", Runtime),
            ):
                export_firmware(model, decision, root / "bundle")
            self.assertEqual(
                (root / "bundle/reference-output.i8").read_bytes(), expected.tobytes()
            )
            self.assertEqual((model / "reference.npz").read_bytes(), original)
            manifest = json.loads((root / "bundle/manifest.json").read_text())
            self.assertEqual(
                manifest["source_reference_max_probability_difference"], 7 / 256
            )

    def test_rejects_mismatched_model_policy_and_frontend_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            content = b"model fixture"
            (model / "model.int8.tflite").write_bytes(content)
            metadata = {
                "frontend": DEFAULT_FRONTEND.to_dict(),
                "labels": list(LABELS),
            }
            selected = {
                "model_sha256": hashlib.sha256(content).hexdigest(),
                "calibration_split": "val",
                "policy": {
                    "threshold": 0.95,
                    "consecutive": 1,
                    "hop_seconds": 0.256,
                    "cooldown_seconds": 3,
                },
            }
            cases = [
                ("model_sha256", "wrong", "Decision settings"),
                ("calibration_split", "test", "Decision settings"),
                ("policy", None, "Invalid"),
                (
                    "frontend",
                    {**metadata["frontend"], "sample_rate": 16000},
                    "frontend",
                ),
                (
                    "policy",
                    {**selected["policy"], "threshold": float("nan")},
                    "Invalid",
                ),
                ("policy", {**selected["policy"], "hop_seconds": 0.00015}, "Invalid"),
                ("policy", {**selected["policy"], "hop_seconds": 0.25}, "align"),
                (
                    "policy",
                    {**selected["policy"], "cooldown_seconds": 2**32},
                    "Invalid",
                ),
            ]
            for key, value, message in cases:
                with self.subTest(key=key, value=value):
                    candidate_metadata = dict(metadata)
                    candidate_policy = dict(selected)
                    (candidate_metadata if key == "frontend" else candidate_policy)[
                        key
                    ] = value
                    (model / "metadata.json").write_text(json.dumps(candidate_metadata))
                    decision = root / "decision.json"
                    decision.write_text(json.dumps(candidate_policy))
                    output = root / "bundle"
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        export_firmware(model, decision, output)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

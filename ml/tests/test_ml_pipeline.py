import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from doorbell_ml.__main__ import parser
from doorbell_ml.benchmark import benchmark
from doorbell_ml.data import LABELS, write_manifest
from doorbell_ml.deployment import export_firmware
from doorbell_ml.evaluate import evaluate, mine
from doorbell_ml.model import Int8Model
from doorbell_ml.training import train


class PipelineTests(unittest.TestCase):
    def test_training_validation_export_and_recording_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for split_index, split in enumerate(("train", "val", "test")):
                rng = np.random.default_rng(split_index)
                for label_index, label in enumerate(LABELS):
                    time = np.arange(24000 if label_index == 0 else 16000) / 8000
                    samples = rng.normal(0, 0.005, len(time))
                    if label_index:
                        frequency = (
                            np.where((time // 0.15) % 2 == 0, 850, 662)
                            if label_index == 1
                            else np.where(time < 0.6, 850, 662)
                        )
                        samples += 0.15 * np.sin(
                            2 * np.pi * np.cumsum(frequency) / 8000
                        )
                    name = f"{split}-{label}.wav"
                    sf.write(root / name, samples, 8000)
                    rows.append(
                        {
                            "path": name,
                            "label": label,
                            "split": split,
                            "group": f"fixture-{split}",
                        }
                    )
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            output = root / "model"
            args = SimpleNamespace(
                manifest=[manifest],
                output=output,
                epochs=2,
                batch_size=4,
                examples_per_class=4,
                validation_per_class=2,
                seed=12,
                synthetic_fraction=0.25,
                background_multiplier=2,
                hard_negatives=None,
                hard_negative_fraction=0.5,
                time_shift_seconds=0.5,
                soundscape_fraction=0.5,
                min_snr_db=-5,
                time_mask_frames=8,
                frequency_mask_bins=1,
                mask_probability=0.5,
                initial_model=None,
                learning_rate=0.001,
                device="cpu",
                max_model_bytes=65536,
                channels=(4, 6, 8),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                metadata = train(args)
            self.assertEqual(metadata["input"]["dtype"], "int8")
            self.assertEqual(metadata["channels"], [4, 6, 8])
            self.assertEqual(metadata["input"]["shape"], [1, 124, 20, 1])
            self.assertEqual(metadata["output"]["dtype"], "int8")
            self.assertLess(metadata["model_bytes"], 65536)
            self.assertFalse(metadata["pico_verified"])
            for changed in (
                metadata | {"input": metadata["input"] | {"scale": 1.0}},
                metadata | {"output": metadata["output"] | {"shape": [1, 4]}},
                metadata | {"labels": list(reversed(LABELS))},
            ):
                with self.subTest(metadata=changed), self.assertRaises(ValueError):
                    Int8Model((output / "model.int8.tflite").read_bytes(), changed)
            self.assertEqual(metadata["augmentation"], "regenerated_each_epoch")
            history = json.loads((output / "history.json").read_text())
            self.assertEqual([row["augmentation_seed"] for row in history], [12, 13])
            self.assertIsNone(metadata["tensor_arena_bytes"])
            self.assertLess(
                metadata["quantization_comparison"]["max_probability_difference"], 0.05
            )
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["synthetic_validation_int8"]["examples"], 6)
            known_manifest = root / "known.csv"
            write_manifest(
                known_manifest,
                [
                    row
                    for row in rows
                    if not (row["split"] == "val" and row["label"] != "background")
                ],
            )
            known_args = SimpleNamespace(
                **(
                    vars(args)
                    | {
                        "manifest": [known_manifest],
                        "output": root / "known-model",
                        "initial_model": output,
                        "learning_rate": 0.0002,
                    }
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                known = train(known_args)
            self.assertEqual(
                known["validation_status"], "known_chimes_heldout_backgrounds"
            )
            known_history = json.loads((known_args.output / "history.json").read_text())
            best = min(known_history, key=lambda row: row["val_loss"])
            self.assertEqual(known["selected_epoch"], best["epoch"])
            self.assertEqual(known["inventory"]["val"]["doorbell_type1"]["files"], 0)
            self.assertIsNotNone(known["initial_model_sha256"])
            self.assertTrue(
                any(
                    row["split"] == "val" and row["label"] == "doorbell_type1"
                    for row in known["sources"]
                )
            )
            evaluation_args = SimpleNamespace(
                manifest=[manifest],
                model=output,
                split="test",
                threshold=0.8,
                hop_seconds=0.25,
                cooldown_seconds=3,
                consecutive=1,
                output=root / "evaluation.json",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                report = evaluate(evaluation_args)
            self.assertEqual(report["clip_metrics"]["examples"], 3)
            self.assertAlmostEqual(report["background_hours"], 3 / 3600)
            self.assertEqual(len(report["recordings"]), 3)
            for recording in report["recordings"]:
                self.assertTrue(
                    all(
                        alarm["seconds"] <= recording["seconds"]
                        for alarm in recording["alarms"]
                    )
                )
            original_report = evaluation_args.output.read_bytes()
            with self.assertRaises(FileExistsError):
                evaluate(evaluation_args)
            self.assertEqual(evaluation_args.output.read_bytes(), original_report)

            with contextlib.redirect_stdout(io.StringIO()):
                mined = mine(
                    parser().parse_args(
                        [
                            "mine",
                            "--manifest",
                            str(manifest),
                            "--model",
                            str(output),
                            "--output",
                            str(root / "hard.json"),
                            "--count",
                            "4",
                        ]
                    )
                )
            self.assertEqual(mined["candidates"], 1)
            self.assertEqual(mined["recordings"][0]["group"], "fixture-train")
            common = [
                "benchmark",
                "--manifest",
                str(manifest),
                "--model",
                str(output),
                "--snr-db",
                "5",
                "--episodes-per-condition",
                "2",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                validation = benchmark(
                    parser().parse_args(
                        common
                        + [
                            "--split",
                            "val",
                            "--calibrate",
                            "--output",
                            str(root / "val"),
                        ]
                    )
                )
                held_out = benchmark(
                    parser().parse_args(
                        common
                        + [
                            "--split",
                            "test",
                            "--policy",
                            str(root / "val/decision.json"),
                            "--output",
                            str(root / "test"),
                        ]
                    )
                )
            self.assertEqual(validation["independent_positive_recordings"], 0)
            self.assertFalse(
                validation["selected"]["known_chime_robustness"][
                    "positive_sources_independent"
                ]
            )
            self.assertEqual(
                held_out["selected"]["policy"], validation["selected"]["policy"]
            )
            self.assertEqual(
                held_out["selected"]["independent_background"]["seconds"], 3
            )
            self.assertEqual(
                held_out["selected"]["known_chime_robustness"]["episodes"], 4
            )
            policy_path = root / "val/decision.json"
            bundle = root / "firmware-model"
            export_firmware(output, policy_path, bundle)
            self.assertEqual(
                (bundle / "model.tflite").read_bytes(),
                (output / "model.int8.tflite").read_bytes(),
            )
            contract = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(contract["policy"], validation["selected"]["policy"])
            self.assertEqual((bundle / "reference-input.i8").stat().st_size, 2480)
            with self.assertRaises(FileExistsError):
                export_firmware(output, policy_path, bundle)
            confirmed = {
                "protocol_version": 4,
                "calibration_split": "val",
                "model_sha256": contract["sha256"]["model.tflite"],
                "replay_sha256": "fixture-replay",
                "decision_sha256": "fixture-decision",
                "sources": metadata["sources"],
                "selection": {"normal_non_regression": True},
                "policy": {
                    "low": 0.6,
                    "high": 0.95,
                    "confirmation_frames": 2,
                    "use_rules": True,
                },
            }
            confirmed_path = root / "confirmed.json"
            confirmed_path.write_text(json.dumps(confirmed))
            export_firmware(output, confirmed_path, root / "confirmed-bundle")
            exported = json.loads((root / "confirmed-bundle/manifest.json").read_text())
            self.assertEqual(exported["policy"]["fusion"], confirmed["policy"])
            self.assertEqual(exported["policy"]["hop_seconds"], 0.256)
            held = confirmed | {
                "protocol_version": 6,
                "policy": confirmed["policy"]
                | {"cooldown_seconds": 10, "rule_hold_ms": 1280},
            }
            held_path = root / "held.json"
            held_path.write_text(json.dumps(held))
            export_firmware(output, held_path, root / "held-bundle")
            held_manifest = json.loads((root / "held-bundle/manifest.json").read_text())
            self.assertEqual(held_manifest["policy"]["cooldown_seconds"], 10)
            self.assertEqual(held_manifest["policy"]["fusion"]["rule_hold_ms"], 1280)
            confirmed["selection"]["normal_non_regression"] = False
            confirmed_path.write_text(json.dumps(confirmed))
            with self.assertRaisesRegex(ValueError, "non-regression"):
                export_firmware(output, confirmed_path, root / "failed-bundle")
            self.assertFalse((root / "failed-bundle").exists())
            policy = json.loads(policy_path.read_text())
            policy["model_sha256"] = "different-model"
            for invalid_policy in (policy, {}, None, []):
                policy_path.write_text(json.dumps(invalid_policy))
                with (
                    self.subTest(policy=invalid_policy),
                    self.assertRaisesRegex(ValueError, "Policy must match"),
                ):
                    benchmark(
                        parser().parse_args(
                            common
                            + [
                                "--split",
                                "test",
                                "--policy",
                                str(policy_path),
                                "--output",
                                str(root / "invalid"),
                            ]
                        )
                    )

            leak = root / "leak.csv"
            write_manifest(
                leak,
                [
                    {
                        "path": "train-background.wav",
                        "label": "background",
                        "split": "test",
                        "group": "different-name",
                    }
                ],
            )
            evaluation_args.manifest = [leak]
            with self.assertRaisesRegex(ValueError, "overlaps model development"):
                evaluate(evaluation_args)


if __name__ == "__main__":
    unittest.main()

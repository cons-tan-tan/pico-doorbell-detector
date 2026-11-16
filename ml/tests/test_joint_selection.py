import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from doorbell_ml.selection import joint_selection_key, select_policy


def candidate(low, false_alarms, detected=10, delay=1.0):
    return {
        "policy": {
            "low": low,
            "high": 0.95,
            "confirmation_frames": 2,
            "minimum_support": 2,
            "use_rules": True,
            "cooldown_seconds": 10.0,
            "rule_hold_ms": 1280,
        },
        "suites": {
            "normal": {
                "background_false_alarms": false_alarms,
                "any_chime_detected": detected,
                "first_correct_kind": detected,
                "detected": detected,
                "wrong_type_alarms": 0,
                "median_presence_delay_seconds": delay,
            }
        },
    }


class JointSelectionTests(unittest.TestCase):
    def test_type_two_gain_cannot_hide_type_one_regression(self):
        reference = candidate(0.8, 0, 18)
        normal = reference["suites"]["normal"]
        normal["conditions"] = [
            {"label": "doorbell_type1", "presence_detected": 10, "detected": 10},
            {"label": "doorbell_type2", "presence_detected": 8, "detected": 8},
        ]
        traded = copy.deepcopy(reference)
        conditions = traded["suites"]["normal"]["conditions"]
        conditions[0] |= {"presence_detected": 8, "detected": 8}
        conditions[1] |= {"presence_detected": 10, "detected": 10}
        self.assertFalse(joint_selection_key([reference], [reference])[0])
        self.assertTrue(joint_selection_key([traded], [reference])[0])

    def test_one_corpus_cannot_hide_regression_in_another(self):
        reference = [candidate(0.6, 1, 9), candidate(0.6, 0, 9)]
        traded = [candidate(0.8, 0, 10, 0.5), candidate(0.8, 1, 10, 0.5)]
        self.assertLess(
            joint_selection_key(reference, reference),
            joint_selection_key(traded, reference),
        )
        traded[1] = candidate(0.8, 0, 8, 0.5)
        self.assertLess(
            joint_selection_key(reference, reference),
            joint_selection_key(traded, reference),
        )

    def test_selects_one_policy_and_rejects_test_or_duplicate_corpora(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, references = [], []
            for index in range(2):
                folder = root / str(index)
                folder.mkdir()
                source = {"path": str(index), "sha256": str(index), "split": "val"}
                protocol = {
                    "version": 7,
                    "model_sha256": "model",
                    "replay_sha256": "replay",
                    "background_sources": [source],
                }
                protocol.update(
                    dict.fromkeys(
                        (
                            "seed",
                            "episodes_per_condition",
                            "snr_db",
                            "suites",
                            "timing",
                            "positive_sources",
                            "warmup_seconds",
                            "post_event_seconds",
                            "insertion_seconds",
                            "speed_factor_range",
                            "rms_dbfs",
                            "snr_definition",
                            "reverb",
                        )
                    )
                )
                reference = {
                    "split": "val",
                    "protocol": protocol,
                    "selected": candidate(0.6, 1, 9),
                }
                report = copy.deepcopy(reference)
                report["validation_candidates"] = [
                    candidate(0.6, index),
                    candidate(0.8, 1 - index),
                    candidate(0.9, 0, 9),
                ]
                paths.append(folder / "report.json")
                paths[-1].write_text(json.dumps(report))
                references.append(folder / "reference.json")
                references[-1].write_text(json.dumps(reference))
                (folder / "decision.json").write_text(
                    json.dumps(
                        {
                            "protocol_version": 7,
                            "calibration_split": "val",
                            "model_sha256": "model",
                            "replay_sha256": "replay",
                            "decision_sha256": "decision",
                            "policy": candidate(0.6, 1)["policy"],
                            "sources": [source],
                        }
                    )
                )
            args = SimpleNamespace(
                reports=paths, references=references, output=root / "selected"
            )
            selected = select_policy(args)
            self.assertEqual(selected["policy"]["low"], 0.9)
            self.assertTrue(selected["selection"]["normal_non_regression"])
            frozen = json.loads((args.output / "decision.json").read_text())
            self.assertEqual(len(frozen["sources"]), 2)
            args.output = root / "invalid"
            report["split"] = "test"
            paths[-1].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "matched validation"):
                select_policy(args)
            self.assertFalse(args.output.exists())
            args.reports = [paths[0], paths[0]]
            args.references = [references[0], references[0]]
            with self.assertRaisesRegex(ValueError, "distinct validation"):
                select_policy(args)


if __name__ == "__main__":
    unittest.main()

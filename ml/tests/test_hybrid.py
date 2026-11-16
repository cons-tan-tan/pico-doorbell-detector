import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from doorbell_ml.audio import Frontend
from doorbell_ml.hybrid import compare, select_threshold
from doorbell_ml.metrics import verify_candidates


class HybridTests(unittest.TestCase):
    def test_empty_policy_cannot_enable_calibration_on_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text(
                json.dumps({"frontend": Frontend().to_dict()})
            )
            (root / "model.int8.tflite").write_bytes(b"model")
            (root / "replay").write_bytes(b"replay")
            (root / "policy.json").write_text("{}")
            args = SimpleNamespace(
                output=root / "output",
                split="test",
                policy=root / "policy.json",
                episodes_per_condition=1,
                snr_db=[5],
                minimum_snr_db=5,
                thresholds=[0.5],
                model=root,
                replay_binary=root / "replay",
            )
            with self.assertRaisesRegex(ValueError, "Policy must match"):
                compare(args)
            self.assertFalse(args.output.exists())

    def verify(self, candidates, scores, times, wait=1.0):
        return verify_candidates(
            candidates, np.asarray(scores), np.asarray(times), 0.8, 0.256, wait, 3.0
        )

    def test_cached_score_does_not_backdate_a_rule_candidate(self):
        alarms = self.verify(
            [{"seconds": 2.1, "label": "doorbell_type1"}],
            [[0.05, 0.9, 0.05]],
            [2.0],
        )
        self.assertEqual(alarms[0]["seconds"], 2.1)

    def test_confirmation_wait_is_counted_in_latency(self):
        alarms = self.verify(
            [{"seconds": 2.1, "label": "doorbell_type1"}],
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]],
            [2.0, 2.512],
        )
        self.assertEqual(alarms[0]["seconds"], 2.512)
        self.assertEqual(alarms[0]["candidate_seconds"], 2.1)

    def test_wrong_type_stale_and_late_scores_do_not_confirm(self):
        alarms = self.verify(
            [{"seconds": 2.1, "label": "doorbell_type1"}],
            [[0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.05, 0.9, 0.05]],
            [1.5, 2.2, 3.2],
        )
        self.assertEqual(alarms, [])

    def test_rejection_does_not_suppress_a_later_valid_candidate(self):
        alarms = self.verify(
            [
                {"seconds": 1.0, "label": "doorbell_type1"},
                {"seconds": 1.5, "label": "doorbell_type1"},
            ],
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]],
            [1.0, 1.5],
            wait=0.1,
        )
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["seconds"], 1.5)

    def test_selection_preserves_rule_detections_before_reducing_false_alarms(self):
        rules = {"episode_details": [{"detected": True, "snr_db": 10}]}
        candidates = [
            {
                "threshold": 0.5,
                "background_false_alarms": 1,
                "wrong_type_alarms": 0,
                "detected": 1,
                "episode_details": [{"detected": True}],
            },
            {
                "threshold": 0.9,
                "background_false_alarms": 0,
                "wrong_type_alarms": 0,
                "detected": 0,
                "episode_details": [{"detected": False}],
            },
        ]
        selected = select_threshold(rules, candidates, 5)
        self.assertEqual(selected["threshold"], 0.5)
        self.assertEqual(selected["lost_rule_detections"], 0)


if __name__ == "__main__":
    unittest.main()

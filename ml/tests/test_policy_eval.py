import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from doorbell_ml.audio import Frontend
from doorbell_ml.metrics import metrics
from doorbell_ml.policy_eval import make_full_episode, selection_key, validate_policy
from doorbell_ml.replay import score_clip


class PolicyEvaluationTests(unittest.TestCase):
    def test_full_recording_and_causal_prediction_trace_are_retained(self):
        config = Frontend()
        positive = SimpleNamespace(path=Path("chime.wav"))
        background = SimpleNamespace(path=Path("background.wav"))
        wave = np.sin(np.arange(5 * config.sample_rate) * 0.7).astype(np.float32)
        noise = (
            np.random.default_rng(0)
            .normal(0, 0.1, config.sample_rate)
            .astype(np.float32)
        )
        with patch(
            "doorbell_ml.policy_eval.load_audio",
            side_effect=lambda path, rate: wave if path == positive.path else noise,
        ):
            normal, onset, offset = make_full_episode(
                positive, background, 10, 42, "normal", config
            )
            repeated, _, _ = make_full_episode(
                positive, background, 10, 42, "normal", config
            )
            reverbed, reverb_onset, reverb_offset = make_full_episode(
                positive, background, 10, 42, "reverb", config
            )
        self.assertAlmostEqual(offset - onset, 5.0)
        self.assertEqual(onset, reverb_onset)
        self.assertGreater(reverb_offset, offset)
        self.assertAlmostEqual(len(normal) / config.sample_rate, offset + 2)
        self.assertGreater(len(reverbed), len(normal))
        np.testing.assert_array_equal(normal, repeated)

        result = SimpleNamespace(
            stdout=b"model,abc\nhop_samples,2048\ntrace_version,2\nprediction,2.000000,127,-128,-128\ntone,2.224000\nprediction,2.256000,-128,64,-64\n"
        )
        with patch("doorbell_ml.replay.subprocess.run", return_value=result):
            row = score_clip(
                Path("replay"),
                np.zeros(2048),
                config,
                {"output": {"zero_point": -128, "scale": 1 / 256}},
                "abc",
            )
        self.assertEqual(row["trace"], ["tone,224", "prediction,256,-128,64,-64"])
        np.testing.assert_allclose(row["times"], [0.256])

    def test_presence_is_distinct_from_correct_type(self):
        background = {"path": "noise", "seconds": 5, "alarms": []}
        episode = {
            "label": "doorbell_type2",
            "snr_db": 10,
            "onset": 1,
            "offset": 3,
            "alarms": [{"seconds": 1.5, "label": "doorbell_type1"}],
        }
        report = metrics(
            [background], [episode], "fusion", 0.6, {}, lambda row: row["alarms"]
        )
        self.assertEqual(report["presence_recall"], 1)
        self.assertEqual(report["recall"], 0)
        self.assertEqual(report["wrong_type_alarms"], 1)
        self.assertEqual(report["median_presence_delay_seconds"], 0.5)
        self.assertEqual(report["first_type_accuracy"], 0)
        episode["alarms"].append({"seconds": 4.5, "label": "doorbell_type2"})
        report = metrics(
            [background], [episode], "fusion", 0.6, {}, lambda row: row["alarms"]
        )
        self.assertEqual(report["recall"], 1)
        self.assertEqual(report["first_type_accuracy"], 0)

    def test_selection_does_not_trade_normal_recall_for_stress_recall(self):
        base = {
            "background_false_alarms": 0,
            "any_chime_detected": 10,
            "detected": 10,
            "wrong_type_alarms": 0,
            "median_audio_delay_seconds": 1.0,
        }
        baseline = {"suites": {"normal": base, "pitch": base.copy()}}
        good = copy.deepcopy(baseline) | {"policy": {"low": 0.8}}
        regressed = copy.deepcopy(good)
        regressed["suites"]["normal"]["any_chime_detected"] = 9
        regressed["suites"]["pitch"]["any_chime_detected"] = 100
        self.assertLess(
            selection_key(good, baseline), selection_key(regressed, baseline)
        )

    def test_frozen_policy_is_bound_to_model_and_both_executables(self):
        frozen = {
            "calibration_split": "val",
            "protocol_version": 4,
            "sources": ["source"],
            "model_sha256": "model",
            "replay_sha256": "replay",
            "decision_sha256": "decision",
            "policy": {
                "low": 0.8,
                "high": 0.95,
                "confirmation_frames": 3,
                "use_rules": True,
            },
        }
        validate_policy(frozen, "model", "replay", "decision")
        timed = frozen | {
            "protocol_version": 5,
            "policy": frozen["policy"] | {"cooldown_seconds": 10},
        }
        validate_policy(timed, "model", "replay", "decision")
        held = timed | {
            "protocol_version": 6,
            "policy": timed["policy"] | {"rule_hold_ms": 1280},
        }
        validate_policy(held, "model", "replay", "decision")
        consistent = held | {
            "protocol_version": 7,
            "policy": held["policy"] | {"minimum_support": 3},
        }
        validate_policy(consistent, "model", "replay", "decision")
        for invalid in (0, 1, 4, 2.5, True):
            with self.assertRaises(ValueError):
                validate_policy(
                    consistent
                    | {"policy": consistent["policy"] | {"minimum_support": invalid}},
                    "model",
                    "replay",
                    "decision",
                )
        for invalid in (-1, 2001, 1.5, True):
            with self.assertRaises(ValueError):
                validate_policy(
                    held | {"policy": held["policy"] | {"rule_hold_ms": invalid}},
                    "model",
                    "replay",
                    "decision",
                )
        for invalid in (-1, float("nan"), True, "10"):
            with self.assertRaises(ValueError):
                validate_policy(
                    timed | {"policy": timed["policy"] | {"cooldown_seconds": invalid}},
                    "model",
                    "replay",
                    "decision",
                )
        for changed in [
            {},
            frozen | {"calibration_split": "test"},
            frozen | {"decision_sha256": "changed"},
            frozen | {"sources": []},
        ]:
            with self.assertRaises(ValueError):
                validate_policy(changed, "model", "replay", "decision")
        with self.assertRaises(ValueError):
            validate_policy(
                frozen | {"policy": frozen["policy"] | {"low": float("nan")}},
                "model",
                "replay",
                "decision",
            )

    def test_accuracy_takes_priority_over_small_latency_gains(self):
        row = {
            "background_false_alarms": 0,
            "any_chime_detected": 10,
            "detected": 10,
            "wrong_type_alarms": 0,
            "median_audio_delay_seconds": 1.5,
            "median_presence_delay_seconds": 1.5,
        }
        baseline = {"suites": {"normal": row}, "policy": {"low": 0.6}}
        faster = copy.deepcopy(baseline)
        faster["suites"]["normal"].update(
            median_presence_delay_seconds=0.8, detected=9, wrong_type_alarms=1
        )
        self.assertLess(
            selection_key(baseline, baseline), selection_key(faster, baseline)
        )
        accurate_and_faster = copy.deepcopy(baseline)
        accurate_and_faster["suites"]["normal"]["median_presence_delay_seconds"] = 0.8
        self.assertLess(
            selection_key(accurate_and_faster, baseline),
            selection_key(baseline, baseline),
        )
        false_alarm = copy.deepcopy(faster)
        false_alarm["suites"]["normal"]["background_false_alarms"] = 1
        self.assertLess(
            selection_key(baseline, baseline), selection_key(false_alarm, baseline)
        )


if __name__ == "__main__":
    unittest.main()

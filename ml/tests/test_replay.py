import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from doorbell_ml.audio import Frontend
from doorbell_ml.replay import run_decisions, score_clip


class ReplayTests(unittest.TestCase):
    output = (
        "model,abc\nhop_samples,2048\ntrace_version,3\n"
        "prediction,2.000000,127,-128,-128\n"
        "pattern,2.224000,doorbell_type1\n"
        "prediction,2.256000,-128,64,-64\n"
        "prediction,2.512000,-128,64,-64\n"
    )

    def score(self, output):
        with patch(
            "doorbell_ml.replay.subprocess.run",
            return_value=SimpleNamespace(stdout=output.encode()),
        ):
            return score_clip(
                Path("replay"),
                np.zeros(4096, dtype=np.float32),
                Frontend(),
                {"output": {"zero_point": -128, "scale": 1 / 256}},
                "abc",
            )

    def test_complete_trace_preserves_prediction_times_and_patterns(self):
        row = self.score(self.output)
        np.testing.assert_allclose(row["times"], [0.256, 0.512])
        np.testing.assert_allclose(row["scores"], [[0, 0.75, 0.25]] * 2)
        self.assertEqual(row["trace"][0], "pattern,224,doorbell_type1")

    def test_incomplete_or_invalid_replay_is_not_scored_as_silence(self):
        for output in (
            self.output.rsplit("prediction,", 1)[0],
            self.output + "prediction,2.512000,-128,64,-64\n",
            self.output.replace("2.256000", "2.257000"),
            self.output.replace("2.256000", "2.100000"),
            self.output.replace("2.256000", "nan"),
            self.output.replace("2.512000", "9.000000"),
            self.output.replace(",64,-64", ",128,-64"),
            self.output.replace("doorbell_type1", "unknown"),
            self.output.replace("trace_version,3", "trace_version,99"),
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.score(output)

    def test_invalid_pcm_is_rejected_before_starting_replay(self):
        for audio in (np.empty(0), np.zeros((1, 4096)), np.array([np.nan])):
            with (
                self.subTest(shape=audio.shape),
                patch("doorbell_ml.replay.subprocess.run") as run,
                self.assertRaisesRegex(ValueError, "finite mono PCM"),
            ):
                score_clip(Path("replay"), audio, Frontend(), {}, "abc")
            run.assert_not_called()

    def test_decision_output_cannot_reassign_clips_or_invent_event_times(self):
        trace = [
            "prediction,256,-128,64,-64",
            "prediction,512,-128,64,-64",
            "prediction,768,-128,64,-64",
        ]
        rows = [
            {"trace_version": 3, "trace": trace, "seconds": 1.0},
            {"trace_version": 3, "trace": trace, "seconds": 1.0},
        ]
        policy = {
            "low": 0.8,
            "high": 0.95,
            "confirmation_frames": 2,
            "use_rules": True,
        }
        for output in (
            "-1,256,doorbell_type1\n",
            "2,256,doorbell_type1\n",
            "0,-1,doorbell_type1\n",
            "0,1001,doorbell_type1\n",
            "0,256,background\n",
            "0,512,doorbell_type1\n0,256,doorbell_type1\n",
        ):
            with (
                self.subTest(output=output),
                patch(
                    "doorbell_ml.replay.subprocess.run",
                    return_value=SimpleNamespace(stdout=output),
                ),
                self.assertRaisesRegex(ValueError, "invalid notification"),
            ):
                run_decisions(Path("decision-replay"), rows, policy)

    def test_cached_decision_trace_cannot_skip_prediction_windows(self):
        row = {
            "trace_version": 3,
            "trace": ["prediction,256,-128,64,-64"],
            "seconds": 1.0,
        }
        with (
            patch("doorbell_ml.replay.subprocess.run") as run,
            self.assertRaisesRegex(ValueError, "complete prediction grid"),
        ):
            run_decisions(Path("decision-replay"), [row], {})
        run.assert_not_called()

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from doorbell_ml import audio as audio_module
from doorbell_ml.audio import Frontend, feature_sequence, features, streaming_features
from doorbell_ml.augmentation import filter_augment
from doorbell_ml.data import file_digest
from doorbell_ml.evaluate import score_audio
from doorbell_ml.selection import (
    NotificationValidator,
    notification_key,
)


class SelectionTests(unittest.TestCase):
    def test_notification_cache_rejects_test_sources_and_training_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "background.wav"
            audio.write_bytes(b"fixture")
            source = {
                "path": str(audio),
                "sha256": file_digest(audio),
                "split": "test",
                "group": "source",
                "label": "background",
            }
            metadata = {
                "split": "val",
                "frontend": Frontend().to_dict(),
                "frontend_source_sha256": file_digest(Path(audio_module.__file__)),
                "sources": [source],
            }
            path = root / "validation.json"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "requires validation backgrounds"):
                NotificationValidator(root, Path("unused"), Frontend(), [])
            source["split"] = "val"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "overlaps training backgrounds"):
                NotificationValidator(
                    root, Path("unused"), Frontend(), [source | {"split": "train"}]
                )
            audio.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source audio changed"):
                NotificationValidator(root, Path("unused"), Frontend(), [])

    def test_pcen_preserves_background_history_between_prediction_windows(self):
        config = Frontend(compression="pcen")
        audio = np.random.default_rng(13).normal(0, 0.1, 48007).astype(np.float32)
        captured = []

        class Runtime:
            def predict(self, inputs):
                captured.extend(inputs)
                return np.zeros((len(inputs), 3))

        result = score_audio(Runtime(), audio, config, 0.256)
        self.assertTrue(np.isfinite(captured).all())
        for index in (0, 1, len(captured) - 1):
            start = (
                round(result["times"][index] * config.sample_rate)
                - config.window_samples
            )
            phase = start % config.hop_samples
            sequence = feature_sequence(audio[phase:], config)
            frame = start // config.hop_samples
            np.testing.assert_allclose(
                captured[index], sequence[frame : frame + 124], atol=1e-6
            )
        reset = features(audio[2048:18048], config)
        self.assertGreater(np.max(np.abs(reset - captured[1])), 0.01)

    def test_filter_gain_uses_power_decibels_and_preserves_calibration(self):
        original = np.full((8, 124, 20, 1), 0.5, dtype=np.float32)
        actual = filter_augment(original, np.random.default_rng(7), 3, 1)
        self.assertTrue(np.all(original == 0.5))
        self.assertGreater(float(np.max(np.abs(actual - original))), 0.01)
        self.assertLessEqual(float(np.max(np.abs(actual - original))), 0.030001)
        np.testing.assert_array_equal(actual[:, 0], actual[:, -1])
        np.testing.assert_array_equal(
            actual, filter_augment(original, np.random.default_rng(7), 3, 1)
        )
        np.testing.assert_array_equal(
            original, filter_augment(original, np.random.default_rng(7), probability=0)
        )

    def test_cached_windows_match_adc_quantized_stream_at_prediction_times(self):
        config = Frontend()
        for length in (0, 1, 2047):
            self.assertEqual(
                streaming_features(np.zeros(length), config).shape,
                (0, *config.shape),
            )
        audio = np.random.default_rng(9).normal(0, 0.1, 48000).astype(np.float32)
        windows = streaming_features(audio, config)
        padded = np.pad(audio, (16000, 0))
        adc = np.clip(np.floor(padded * 2048 + 2048.5), 0, 4095)
        quantized = (adc - 2048) / 2048
        self.assertEqual(len(windows), len(audio) // 2048)
        for index in (0, 1, 7, len(windows) - 1):
            start = (index + 1) * 2048
            expected = features(quantized[start : start + 16000], config)
            np.testing.assert_allclose(windows[index], expected, atol=1e-6)

    def test_notification_selection_prioritizes_accuracy_over_speed(self):
        suite = {
            "background_false_alarms": 0,
            "presence_recall": 0.95,
            "any_chime_detected": 190,
            "wrong_type_alarms": 0,
            "first_correct_kind": 190,
            "median_presence_delay_seconds": 2.0,
        }
        key = notification_key({"suites": {"normal": suite}})
        for worse in (
            {"background_false_alarms": 1, "any_chime_detected": 200},
            {"presence_recall": 0.94, "any_chime_detected": 188},
            {"wrong_type_alarms": 1},
        ):
            self.assertLess(
                key,
                notification_key(
                    {
                        "suites": {
                            "normal": suite
                            | worse
                            | {"median_presence_delay_seconds": 0.1}
                        }
                    }
                ),
            )

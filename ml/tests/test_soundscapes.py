import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from doorbell_ml.audio import Frontend
from doorbell_ml.augmentation import spec_augment
from doorbell_ml.data import (
    LABELS,
    Recording,
    file_digest,
    load_hard_negatives,
    make_examples,
)
from doorbell_ml.metrics import notification_metrics
from doorbell_ml.soundscapes import SoundscapeCorpus, stream_windows


class SoundscapeTests(unittest.TestCase):
    def test_scaper_is_repeatable_and_aligns_annotations_with_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            time = np.arange(16000) / 8000
            records = []
            for label, frequency in zip(LABELS, (200, 850, 662), strict=True):
                path = root / f"{label}.wav"
                sf.write(path, 0.1 * np.sin(2 * np.pi * frequency * time), 8000)
                records.append(Recording(path, label, "train", label))
            before = [file_digest(record.path) for record in records]
            with SoundscapeCorpus(records, Frontend()) as corpus:
                first, events = corpus.render(
                    records[0], [(records[1], 3.0, 10)], np.random.default_rng(7)
                )
                second, other = corpus.render(
                    records[0], [(records[1], 3.0, 10)], np.random.default_rng(7)
                )
            np.testing.assert_array_equal(first, second)
            self.assertEqual(events, other)
            self.assertEqual(
                events, [{"onset": 3.0, "offset": 5.0, "label": LABELS[1]}]
            )
            self.assertEqual(len(first), 96000)
            self.assertLessEqual(np.max(np.abs(first)), 1)
            frequencies = np.fft.rfftfreq(8000, 1 / 8000)
            peak_before = frequencies[np.abs(np.fft.rfft(first[8000:16000])).argmax()]
            peak_during = frequencies[np.abs(np.fft.rfft(first[28000:36000])).argmax()]
            self.assertEqual(peak_before, 200)
            self.assertEqual(peak_during, 850)
            self.assertEqual(before, [file_digest(record.path) for record in records])
            self.assertNotIn("Inf", vars(np))

    def test_partial_chimes_are_not_silently_labelled_background(self):
        events = [{"onset": 3.0, "offset": 5.0, "label": LABELS[1]}]
        windows, labels, times = stream_windows(
            np.ones(64000, dtype=np.float32), events, Frontend()
        )
        self.assertEqual(windows.shape[1], 16000)
        self.assertTrue(np.all(labels[(times <= 3) | (times >= 7)] == 0))
        self.assertFalse(((times > 3) & (times < 3.75)).any())
        self.assertTrue(np.all(labels[(times >= 3.75) & (times <= 6.25)] == 1))
        np.testing.assert_allclose((times - 2) / 0.256, np.rint((times - 2) / 0.256))

    def test_specaugment_does_not_mutate_calibration_and_keeps_masks_small(self):
        original = np.arange(4 * 124 * 20, dtype=np.float32).reshape(4, 124, 20, 1)
        saved = original.copy()
        masked = spec_augment(original, np.random.default_rng(19), 8, 1, 1)
        np.testing.assert_array_equal(original, saved)
        np.testing.assert_array_equal(
            masked, spec_augment(original, np.random.default_rng(19), 8, 1, 1)
        )
        counts = (masked != original).sum(axis=(1, 2, 3))
        self.assertTrue((counts > 0).any())
        self.assertTrue((counts <= 8 * 20 + 124).all())
        np.testing.assert_array_equal(
            original, spec_augment(original, np.random.default_rng(19), probability=0)
        )

    def test_hard_negative_keeps_the_mined_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for index, label in enumerate(LABELS):
                path = root / f"{label}.wav"
                samples = np.random.default_rng(index).normal(0, 0.01, 48000)
                if index == 0:
                    samples[:16000] = 0
                sf.write(path, samples, 8000)
                records.append(Recording(path, label, "train", label))
            report = {
                "split": "train",
                "sample_rate": 8000,
                "window_samples": 16000,
                "recordings": [
                    {
                        "path": str(records[0].path),
                        "group": records[0].group,
                        "sha256": file_digest(records[0].path),
                        "start_sample": 0,
                    }
                ],
            }
            path = root / "hard.json"
            path.write_text(json.dumps(report))
            hard = load_hard_negatives(path, records)
            _, labels, pcm = make_examples(
                records,
                "train",
                20,
                42,
                Frontend(),
                synthetic_fraction=0,
                hard_negatives=hard,
                hard_negative_fraction=1,
            )
            self.assertGreater(
                np.sum(np.max(np.abs(pcm[labels == 0]), axis=1) == 0), 10
            )
            report["recordings"][0]["start_sample"] = 48000
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Invalid hard-negative window"):
                load_hard_negatives(path, records)

    def test_notification_matching_counts_duplicates_wrong_kinds_and_false_alarms(self):
        events = [
            {"onset": 2, "offset": 4, "label": LABELS[1]},
            {"onset": 10, "offset": 12, "label": LABELS[2]},
        ]
        alarms = [
            {"seconds": 1, "label": LABELS[1]},
            {"seconds": 3, "label": LABELS[2]},
            {"seconds": 5, "label": LABELS[1]},
            {"seconds": 11, "label": LABELS[2]},
        ]
        result = notification_metrics(events, alarms, 20)
        self.assertEqual(result["detected"], 2)
        self.assertEqual(result["first_kind_correct"], 1)
        self.assertEqual(result["wrong_kind_notifications"], 1)
        self.assertEqual(result["duplicate_notifications"], 1)
        self.assertEqual(result["false_notifications"], 1)

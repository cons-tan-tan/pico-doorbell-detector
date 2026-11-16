import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from doorbell_ml.audio import (
    Frontend,
    crop_event,
    features,
    load_audio,
    mix_at_snr,
    rms,
    shift_event,
)
from doorbell_ml.benchmark import benchmark, make_episode, select_policy
from doorbell_ml.data import (
    Recording,
    choose_positive,
    file_digest,
    load_hard_negatives,
    make_examples,
    read_manifests,
    write_manifest,
)
from doorbell_ml.evaluate import score_audio, window_starts
from doorbell_ml.metrics import alarm_times
from doorbell_ml.sources import import_esc50
from doorbell_ml.synth import make_midi

ROOT = Path(__file__).resolve().parents[2]


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for index in range(3):
            sf.write(
                self.root / f"{index}.wav",
                np.random.default_rng(index).normal(0, 0.01, 8000),
                8000,
            )

    def rows(self):
        return [
            {
                "path": "0.wav",
                "label": "background",
                "split": "train",
                "group": "session-a",
            },
            {
                "path": "1.wav",
                "label": "background",
                "split": "test",
                "group": "session-b",
            },
        ]

    def test_same_session_cannot_cross_splits(self):
        rows = self.rows()
        rows[1]["group"] = "session-a"
        path = self.root / "manifest.csv"
        write_manifest(path, rows)
        with self.assertRaisesRegex(ValueError, "appears in both"):
            read_manifests([path])

    def test_repeated_clips_do_not_outweigh_another_positive_session(self):
        records = [
            Recording(Path(f"{index}.wav"), "doorbell_type2", "train", group)
            for index, group in enumerate(["old"] + ["new"] * 8)
        ]
        rng = np.random.default_rng(42)
        chosen = [choose_positive(records, rng) for _ in range(4000)]
        old_fraction = sum(record.group == "old" for record in chosen) / len(chosen)
        self.assertTrue(0.47 < old_fraction < 0.53)
        self.assertEqual(set(chosen), set(records))

    def test_renamed_duplicate_cannot_cross_splits(self):
        (self.root / "1.wav").write_bytes((self.root / "0.wav").read_bytes())
        path = self.root / "manifest.csv"
        write_manifest(path, self.rows())
        with self.assertRaisesRegex(ValueError, "Duplicate audio"):
            read_manifests([path])

    def test_synthetic_audio_cannot_be_held_out_real_audio(self):
        rows = self.rows()
        rows[1]["kind"] = "synthetic"
        path = self.root / "manifest.csv"
        write_manifest(path, rows)
        with self.assertRaisesRegex(ValueError, "Synthetic instruments"):
            read_manifests([path])

    def test_training_examples_are_repeatable_and_keep_source_audio(self):
        records = read_manifests([ROOT / "data/recordings.csv"])
        source = load_audio(records[0].path).copy()
        first = make_examples(records, "train", 3, 42, Frontend())
        second = make_examples(records, "train", 3, 42, Frontend())
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(source, load_audio(records[0].path))
        self.assertEqual(first[0].shape, (9, 124, 20, 1))
        np.testing.assert_array_equal(np.bincount(first[1]), [3, 3, 3])
        self.assertTrue(np.isfinite(first[0]).all())

    def test_esc50_preserves_original_source_and_folds(self):
        (self.root / "audio").mkdir()
        (self.root / "meta").mkdir()
        (self.root / "LICENSE").write_text("test fixture")
        with (self.root / "meta/esc50.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("filename", "fold", "category", "src_file", "esc10")
            )
            writer.writeheader()
            for fold in range(1, 6):
                name = f"{fold}.wav"
                (self.root / "audio" / name).write_bytes(
                    (self.root / f"{fold % 3}.wav").read_bytes()
                )
                writer.writerow(
                    {
                        "filename": name,
                        "fold": fold,
                        "category": "rain",
                        "src_file": str(fold),
                        "esc10": "True",
                    }
                )
        output = self.root / "manifest.csv"
        import_esc50(
            SimpleNamespace(
                root=self.root, output=output, categories=["rain"], esc10_only=True
            )
        )
        with output.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [row["split"] for row in rows], ["train", "train", "train", "val", "test"]
        )
        self.assertEqual(rows[-1]["group"], "freesound:5")
        self.assertIn("CC-BY-3.0", rows[0]["license"])

    def test_epoch_variation_background_balance_and_quiet_examples(self):
        records = read_manifests([ROOT / "data/recordings.csv"])
        first = make_examples(
            records, "train", 64, 42, Frontend(), background_multiplier=2
        )
        second = make_examples(
            records, "train", 64, 43, Frontend(), background_multiplier=2
        )
        np.testing.assert_array_equal(np.bincount(first[1]), [128, 64, 64])
        self.assertFalse(np.array_equal(first[0], second[0]))
        negative_levels = np.array([rms(pcm) for pcm in first[2][first[1] == 0]])
        self.assertTrue((negative_levels < 1e-4).any())
        self.assertTrue((negative_levels > 0.05).any())

    def test_esc50_moves_overlapping_source_folds_to_test(self):
        (self.root / "audio").mkdir()
        (self.root / "meta").mkdir()
        (self.root / "LICENSE").write_text("test fixture")
        (self.root / "meta/esc50.csv").write_text(
            "filename,fold,category,src_file,esc10\n"
            "0.wav,4,clock_alarm,209698,False\n"
            "1.wav,5,clock_tick,209698,True\n"
        )
        for name in ("0.wav", "1.wav"):
            (self.root / "audio" / name).write_bytes((self.root / name).read_bytes())
        output = self.root / "manifest.csv"
        import_esc50(
            SimpleNamespace(
                root=self.root,
                output=output,
                categories=[],
                all_categories=True,
                esc10_only=False,
            )
        )
        records = read_manifests([output])
        self.assertEqual([r.split for r in records], ["test", "test"])

    def test_hard_negatives_reject_validation_sources_and_changed_files(self):
        manifest = self.root / "manifest.csv"
        write_manifest(manifest, self.rows())
        records = read_manifests([manifest])
        report_path = self.root / "hard.json"
        row = {
            "path": str(records[0].path),
            "group": records[0].group,
            "sha256": file_digest(records[0].path),
        }
        report_path.write_text(json.dumps({"split": "train", "recordings": [row]}))
        self.assertEqual(
            load_hard_negatives(report_path, records)[0].recording, records[0]
        )
        row["path"] = str(records[1].path)
        report_path.write_text(json.dumps({"split": "train", "recordings": [row]}))
        with self.assertRaisesRegex(ValueError, "recorded training backgrounds"):
            load_hard_negatives(report_path, records)
        row["path"] = str(records[0].path)
        row["sha256"] = "changed"
        report_path.write_text(json.dumps({"split": "train", "recordings": [row]}))
        with self.assertRaisesRegex(ValueError, "recorded training backgrounds"):
            load_hard_negatives(report_path, records)


class AudioTests(unittest.TestCase):
    def test_streaming_scores_only_complete_windows_on_the_prediction_grid(self):
        class Runtime:
            def predict(self, windows):
                return np.zeros((len(windows), 3), dtype=np.float32)

        result = score_audio(
            Runtime(),
            np.zeros(40000, dtype=np.float32),
            Frontend(),
            0.256,
            streaming=True,
        )
        np.testing.assert_allclose(result["times"], 2 + np.arange(12) * 0.256)
        self.assertEqual(result["scores"].shape, (12, 3))
        short = score_audio(
            Runtime(),
            np.zeros(4000, dtype=np.float32),
            Frontend(),
            0.256,
            streaming=True,
        )
        self.assertEqual(short["scores"].shape, (0, 3))
        self.assertEqual(len(short["times"]), 0)

    def test_frontend_removes_dc_and_handles_silence(self):
        np.testing.assert_array_equal(
            features(np.ones(16000, dtype=np.float32)), np.zeros((124, 20, 1))
        )
        time = np.arange(16000) / 8000
        wave = (0.1 * np.sin(2 * np.pi * 850 * time)).astype(np.float32)
        np.testing.assert_allclose(features(wave), features(wave + 0.3), atol=2e-5)

    def test_positive_crop_does_not_choose_long_silence(self):
        samples = np.zeros(80000, dtype=np.float32)
        samples[60000:64000] = np.sin(2 * np.pi * 850 * np.arange(4000) / 8000)
        for seed in range(8):
            crop = crop_event(samples, 16000, np.random.default_rng(seed))
            self.assertGreater(float(np.sum(crop**2)), 1900)
        with self.assertRaisesRegex(ValueError, "silent"):
            crop_event(np.zeros(16000), 16000, np.random.default_rng(0))

    def test_snr_is_measured_on_active_foreground(self):
        foreground = np.concatenate((np.zeros(8000), np.ones(8000))).astype(np.float32)
        noise = np.ones(16000, dtype=np.float32) * 0.5
        mixture = mix_at_snr(foreground, noise, 20)
        self.assertAlmostEqual(
            20 * np.log10(rms(foreground[8000:]) / rms(mixture - foreground)),
            20,
            places=4,
        )

    def test_time_shift_retains_event_energy_without_wraparound(self):
        source = np.linspace(0.1, 1, 16000, dtype=np.float32)
        changed = False
        for seed in range(10):
            shifted = shift_event(source, 4000, np.random.default_rng(seed))
            self.assertGreaterEqual(float(np.sum(shifted**2)), 0.7 * np.sum(source**2))
            if not np.array_equal(source, shifted):
                changed = True
                self.assertTrue(shifted[0] == 0 or shifted[-1] == 0)
        self.assertTrue(changed)
        np.testing.assert_array_equal(
            source, shift_event(source, 0, np.random.default_rng(1))
        )

    def test_cooldown_and_background_decisions(self):
        probabilities = np.array(
            [[0.1, 0.9, 0], [0.1, 0.9, 0], [0.9, 0.1, 0], [0.1, 0, 0.9]]
        )
        alarms = alarm_times(probabilities, np.array([2, 2.25, 5, 6]), 0.8, 3)
        self.assertEqual([alarm["seconds"] for alarm in alarms], [2, 6])
        self.assertEqual(window_starts(17000, 16000, 2000), [0, 1000])
        self.assertEqual(window_starts(4000, 16000, 2000), [0])

    def test_midi_contains_balanced_notes_and_selected_instrument(self):
        import mido

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.mid"
            make_midi(path, np.random.default_rng(7), 24)
            messages = list(mido.MidiFile(path))
        self.assertEqual(
            next(
                message.program
                for message in messages
                if message.type == "program_change"
            ),
            24,
        )
        active = set()
        for message in messages:
            if message.type == "note_on":
                self.assertNotIn(message.note, active)
                active.add(message.note)
            elif message.type == "note_off":
                active.remove(message.note)
        self.assertFalse(active)

    def test_consecutive_predictions_reset_on_background_and_other_chimes(self):
        labels = [1, 0, 1, 2, 2, 0, 1, 1]
        probabilities = np.eye(3)[labels]
        alarms = alarm_times(probabilities, np.arange(8) * 0.25, 0.8, 0, consecutive=2)
        self.assertEqual([a["seconds"] for a in alarms], [1.0, 1.75])
        self.assertEqual(
            [a["label"] for a in alarms], ["doorbell_type2", "doorbell_type1"]
        )

    def test_episode_is_repeatable_without_modifying_recordings(self):
        records = read_manifests([ROOT / "data/recordings.csv"])
        source = load_audio(records[0].path).copy()
        first = make_episode(
            records[0], records[2], 5, np.random.default_rng(2), Frontend()
        )
        second = make_episode(
            records[0], records[2], 5, np.random.default_rng(2), Frontend()
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(source, load_audio(records[0].path))
        self.assertEqual(first[0].shape, (40000,))
        self.assertGreaterEqual(first[1], 0.75)
        self.assertLess(first[1], 1.75)
        self.assertAlmostEqual(first[2] - first[1], 2)


class PolicyTests(unittest.TestCase):
    def test_policy_selection_retains_required_recall_before_reducing_alarms(self):
        def candidate(recall, false_alarms, threshold):
            return {
                "policy": {"threshold": threshold, "consecutive": 1},
                "independent_background": {"false_alarms": false_alarms},
                "known_chime_robustness": {
                    "conditions": [
                        {"snr_db": 5, "recall": recall, "wrong_type_alarms": 0}
                    ]
                },
            }

        candidates = [
            candidate(0.7, 0, 0.98),
            candidate(1, 5, 0.5),
            candidate(0.95, 1, 0.8),
        ]
        selected, status = select_policy(candidates, 0.95, 5)
        self.assertEqual(selected["policy"]["threshold"], 0.8)
        self.assertTrue(status["target_met"])
        _, status = select_policy([candidates[0]], 0.95, 5)
        self.assertFalse(status["target_met"])

    def test_calibration_cannot_use_test_split(self):
        with self.assertRaisesRegex(ValueError, "only on val"):
            benchmark(SimpleNamespace(calibrate=True, split="test"))


if __name__ == "__main__":
    unittest.main()

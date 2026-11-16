import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from doorbell_ml.__main__ import parser
from doorbell_ml.data import read_manifests
from doorbell_ml.prepare import (
    MultipartZipReader,
    fetch_esc,
    fetch_fsd,
    fetch_range,
    store_audio,
)
from doorbell_ml.retrain import recipe_commands

ROOT = Path(__file__).resolve().parents[2]


class ReproducibilityTests(unittest.TestCase):
    def test_default_inputs_resolve_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            cli = parser()
            data = cli.parse_args(["prepare-data"]).data
            train = cli.parse_args(["train", "--output", "new-model"])
            recipe = cli.parse_args(["retrain", "--output", "new-run"]).recipe
            manifests, _ = recipe_commands(recipe, Path(temporary) / "new-run")
            self.assertEqual(data, ROOT / "data")
            self.assertEqual(train.manifest, [data / "recordings.csv"])
            self.assertEqual(recipe, ROOT / "ml/training.toml")
            self.assertTrue(all(path.is_file() for path in manifests))

    def test_recording_inventory_is_ready_to_train_and_checked(self):
        import soundfile as sf

        records = read_manifests([ROOT / "data/recordings.csv"])
        provenance = json.loads((ROOT / "data/recordings/provenance.json").read_text())
        self.assertEqual(len(records), len(provenance["clips"]))
        self.assertFalse(provenance["originals_retained"])
        for record in records:
            info = sf.info(record.path)
            self.assertEqual(
                (info.samplerate, info.channels, info.subtype), (8000, 1, "FLOAT")
            )
            self.assertTrue(record.path.is_relative_to(ROOT / "data/recordings"))

    def test_manifest_checksum_detects_changed_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip = root / "clip.wav"
            clip.write_bytes(b"changed")
            with (root / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("path", "label", "split", "group", "sha256")
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "path": clip.name,
                        "label": "background",
                        "split": "train",
                        "group": "g",
                        "sha256": "0" * 64,
                    }
                )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_manifests([root / "manifest.csv"])

    def test_failed_download_does_not_replace_existing_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.wav"
            path.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                store_audio(path, b"wrong", "0" * 64)
            self.assertEqual(path.read_bytes(), b"original")
            store_audio(path, b"verified", hashlib.sha256(b"verified").hexdigest())
            self.assertEqual(path.read_bytes(), b"verified")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_range_response_must_match_exact_request(self):
        for status, content_range, payload in (
            (200, "bytes 2-4/8", b"cde"),
            (206, "bytes 1-3/8", b"bcd"),
            (206, "bytes 2-4/8", b"cd"),
        ):
            response = io.BytesIO(payload)
            response.status = status
            response.headers = {"Content-Range": content_range}
            with (
                patch("urllib.request.urlopen", return_value=response),
                self.assertRaises(ValueError),
            ):
                fetch_range("https://example.invalid/data", 2, 4, 8)
        response = io.BytesIO(b"cde")
        response.status = 206
        response.headers = {"Content-Range": "bytes 2-4/8"}
        with patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(
                fetch_range("https://example.invalid/data", 2, 4, 8), b"cde"
            )

    def test_multipart_zip_reads_across_volume_boundary(self):
        parts = [{"url": "a", "size": 4}, {"url": "b", "size": 4}]
        contents = {"a": b"abcd", "b": b"efgh"}

        def fetch(url, start, end, total):
            self.assertEqual(total, len(contents[url]))
            return contents[url][start : end + 1]

        with patch("doorbell_ml.prepare.fetch_range", side_effect=fetch):
            reader = MultipartZipReader(parts)
            reader.seek(2)
            self.assertEqual(reader.read(4), b"cdef")
            reader.seek(-3, 2)
            self.assertEqual(reader.read(), b"fgh")
            self.assertEqual(reader.read(), b"")

    def test_fsd_download_extracts_only_locked_member(self):
        content = b"fixture audio" * 20
        zipped = io.BytesIO()
        with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("FSD50K.eval_audio/1.wav", content)
            archive.writestr("FSD50K.eval_audio/2.wav", b"not selected")
        payload = zipped.getvalue()
        parts = [{"url": "a", "size": len(payload)}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "1.wav"
            with (
                patch("doorbell_ml.prepare.fsd_directory", return_value=payload),
                patch(
                    "doorbell_ml.prepare.fetch_range",
                    side_effect=lambda url, start, end, total: payload[start : end + 1],
                ),
            ):
                fetch_fsd(parts, [(path, hashlib.sha256(content).hexdigest())])
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_esc_download_extracts_only_locked_member(self):
        zipped = io.BytesIO()
        content = b"fixture audio"
        with zipfile.ZipFile(zipped, "w") as archive:
            archive.writestr("revision/audio/1.wav", content)
            archive.writestr("../../unselected.wav", b"not selected")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "1.wav"
            with patch(
                "urllib.request.urlopen", return_value=io.BytesIO(zipped.getvalue())
            ):
                fetch_esc(
                    {
                        "archive": "https://example.invalid/a.zip",
                        "prefix": "revision/audio/",
                    },
                    [(path, hashlib.sha256(content).hexdigest())],
                )
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_recipe_initializes_then_mines_and_refines_its_own_model(self):
        output = Path("/tmp/new-run")
        manifests, commands = recipe_commands(ROOT / "ml/training.toml", output)
        self.assertEqual(len(manifests), 4)
        self.assertNotIn("--initial-model", commands[0])
        self.assertEqual(
            commands[1][commands[1].index("--model") + 1], str(output / "base")
        )
        self.assertEqual(
            commands[2][commands[2].index("--initial-model") + 1], str(output / "base")
        )
        self.assertEqual(
            commands[2][commands[2].index("--hard-negatives") + 1],
            str(output / "hard-negatives.json"),
        )
        self.assertTrue(all("artifacts/" not in str(path) for path in manifests))

    def test_recipe_rejects_hidden_dependency_on_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            recipe = Path(temporary) / "recipe.toml"
            recipe.write_text(
                (ROOT / "ml/training.toml")
                .read_text()
                .replace("[base]", '[base]\ninitial_model = "old-model"')
            )
            with self.assertRaisesRegex(ValueError, "previous runs"):
                recipe_commands(recipe, Path(temporary) / "output")


if __name__ == "__main__":
    unittest.main()

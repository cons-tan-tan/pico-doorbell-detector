import unittest

import numpy as np

from doorbell_ml.sources import select_fsd50k


class AccuracyTests(unittest.TestCase):
    def test_background_selection_excludes_target_labels_and_related_uploaders(self):
        metadata = {
            str(index): {
                "uploader": f"person-{index // 2}",
                "license": "CC0",
                "title": "sound",
                "tags": [],
            }
            for index in range(30)
        }
        records = [{"fname": key, "labels": "Guitar"} for key in metadata]
        records[3]["labels"] = "Doorbell,Bell"
        metadata["5"]["title"] = "a doorbell"
        selected = select_fsd50k(records, metadata, {"0"}, [30, 30, 30], 19)
        self.assertTrue(
            {"0", "1", "3", "5"}.isdisjoint({row["fname"] for row in selected})
        )
        groups = {}
        for row in selected:
            uploader = metadata[row["fname"]]["uploader"]
            self.assertEqual(groups.setdefault(uploader, row["split"]), row["split"])
        self.assertEqual(
            selected,
            select_fsd50k(list(reversed(records)), metadata, {"0"}, [30, 30, 30], 19),
        )

    def test_distillation_zero_weight_matches_supervised_loss_and_gradient_is_finite(
        self,
    ):
        from doorbell_ml.distillation import distillation_loss
        from doorbell_ml.training import configure_tensorflow

        tf = configure_tensorflow("cpu", 12)
        targets = tf.constant([[0, 1, 0, 0.1, 0.8, 0.1]], tf.float32)
        logits = tf.Variable([[0.1, 0.2, 0.3]], dtype=tf.float32)
        probabilities = tf.nn.softmax(logits)
        expected = tf.keras.losses.categorical_crossentropy(
            targets[:, :3], probabilities
        )
        np.testing.assert_allclose(
            distillation_loss(0, 2)(targets, probabilities), expected
        )
        with tf.GradientTape() as tape:
            loss = distillation_loss(0.25, 2)(targets, tf.nn.softmax(logits))
        gradient = tape.gradient(loss, logits).numpy()
        self.assertTrue(np.isfinite(gradient).all())
        self.assertLess(gradient[0, 1], 0)
        self.assertGreater(gradient[0, 0], 0)

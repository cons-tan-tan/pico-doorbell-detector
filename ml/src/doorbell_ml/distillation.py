import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import resample_poly

from .audio import Frontend, load_audio
from .data import (
    LABELS,
    file_digest,
    load_hard_negatives,
    make_examples,
    provenance,
    read_manifests,
    write_json,
)
from .model import float_predictions
from .policy_eval import make_full_episode
from .selection import NotificationValidator, notification_key
from .training import configure_tensorflow


class AudioEmbeddings:
    def __init__(self, pretrained: Path):
        import tensorflow as tf

        self.pretrained = tf.saved_model.load(str(pretrained))
        shape = self.pretrained(tf.zeros(32000, tf.float32)).shape
        if tuple(shape) != (4, 1024):
            raise ValueError("Expected four YAMNet embeddings per two-second waveform")

        @tf.function(input_signature=[tf.TensorSpec([None, 32000], tf.float32)])
        def extract(inputs):
            return tf.map_fn(
                self.pretrained,
                inputs,
                fn_output_signature=tf.TensorSpec([4, 1024], tf.float32),
            )

        self.extract = extract

    def predict(self, pcm):
        if pcm.ndim != 2 or pcm.shape[1] != 16000 or not np.isfinite(pcm).all():
            raise ValueError("Teacher expects finite two-second, 8000-Hz waveforms")
        results = []
        for start in range(0, len(pcm), 16):
            samples = resample_poly(pcm[start : start + 16], 2, 1, axis=1).astype(
                np.float32
            )
            results.append(self.extract(samples).numpy())
        return np.concatenate(results)


class Teacher:
    def __init__(self, directory: Path):
        import tensorflow as tf

        metadata = json.loads((directory / "metadata.json").read_text())
        if (
            metadata["labels"] != list(LABELS)
            or metadata["teacher_type"] != "yamnet_temporal_head"
        ):
            raise ValueError("Unsupported teacher model")
        pretrained = Path(metadata["pretrained"])
        if file_digest(pretrained / "saved_model.pb") != metadata["pretrained_sha256"]:
            raise ValueError("Pretrained teacher changed")
        self.embeddings = AudioEmbeddings(pretrained)
        self.head = tf.keras.models.load_model(directory / "model.keras", compile=False)

    def predict(self, pcm):
        return float_predictions(self.head, self.embeddings.predict(pcm))


def distillation_loss(weight, temperature):
    import tensorflow as tf

    if not 0 <= weight < 1 or not 1 <= temperature <= 10:
        raise ValueError("Invalid distillation weight or temperature")

    def loss(targets, prediction):
        hard, teacher = targets[:, :3], targets[:, 3:]
        teacher = tf.nn.softmax(tf.math.log(tf.maximum(teacher, 1e-7)) / temperature)
        student = tf.nn.softmax(tf.math.log(tf.maximum(prediction, 1e-7)) / temperature)
        supervised = tf.keras.losses.categorical_crossentropy(hard, prediction)
        soft = tf.keras.losses.kullback_leibler_divergence(teacher, student)
        return (1 - weight) * supervised + weight * temperature**2 * soft

    return loss


def validation_embeddings(validator, extractor, output):
    inputs = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(len(validator.x), 4, 1024)
    )
    config = Frontend()
    for index, row in enumerate(validator.rows):
        if "path" in row:
            audio = load_audio(Path(row["path"]))
        else:
            audio, _, _ = make_full_episode(
                SimpleNamespace(path=Path(row["positive"])),
                SimpleNamespace(path=Path(row["background"])),
                row["snr_db"],
                row["augmentation_seed"],
                row["suite"],
                config,
            )
        pcm = np.pad(audio, (config.window_samples, 0))
        adc = np.clip(np.floor(pcm * 2048 + 2048.5), 0, 4095)
        pcm = (adc - 2048).astype(np.float32) / 2048
        windows = np.stack(
            [pcm[start : start + 16000] for start in range(2048, len(audio) + 1, 2048)]
        )
        start = row["feature_start"]
        inputs[start : start + len(windows)] = extractor.predict(windows)
        if index % 100 == 0:
            print(
                f"Teacher validation embeddings: {index}/{len(validator.rows)} clips",
                flush=True,
            )
    inputs.flush()
    return inputs


def train_teacher(args):
    if min(args.pools, args.epochs, args.examples_per_class) < 1:
        raise ValueError("Teacher pools, epochs and example counts must be positive")
    records = read_manifests(args.manifest)
    sources = provenance(records)
    config = Frontend()
    validator = NotificationValidator(
        args.notification_validation, args.decision_binary, config, sources
    )
    tf = configure_tensorflow("cpu", args.seed)
    extractor = AudioEmbeddings(args.pretrained)
    args.output.mkdir(parents=True, exist_ok=False)
    hard = load_hard_negatives(args.hard_negatives, records)
    embeddings, labels = [], []
    for index in range(args.pools):
        _, y, pcm = make_examples(
            records,
            "train",
            args.examples_per_class,
            args.seed + index,
            config,
            background_multiplier=4,
            hard_negatives=hard,
            hard_negative_fraction=0.6,
            soundscape_fraction=0.25,
            min_snr_db=-5,
        )
        embeddings.append(extractor.predict(pcm))
        labels.append(y)
        print(
            f"Teacher training embeddings: {index + 1}/{args.pools} pools", flush=True
        )
    x, y = np.concatenate(embeddings), np.concatenate(labels)
    np.savez_compressed(args.output / "training.npz", x=x, y=y)
    vx = validation_embeddings(validator, extractor, args.output / "validation.npy")
    model = tf.keras.Sequential(
        [
            tf.keras.Input((4, 1024)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0003),
        loss="sparse_categorical_crossentropy",
        jit_compile=False,
    )
    rng = np.random.default_rng(args.seed)
    best, history, selected = None, [], None
    for epoch in range(args.epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 64):
            chosen = order[start : start + 64]
            model.train_on_batch(x[chosen], y[chosen])
        if (epoch + 1) % 4 == 0 or epoch + 1 == args.epochs:
            summary = validator.evaluate(model, vx)
            row = {"epoch": epoch + 1, "notifications": summary}
            history.append(row)
            print(json.dumps(row), flush=True)
            key = notification_key(summary)
            if best is None or key < best:
                best, selected = key, row
                model.save(args.output / "model.keras")
    write_json(args.output / "history.json", history)
    write_json(
        args.output / "metadata.json",
        {
            "teacher_type": "yamnet_temporal_head",
            "labels": list(LABELS),
            "sources": sources,
            "seed": args.seed,
            "pools": args.pools,
            "examples_per_class": args.examples_per_class,
            "pretrained": str(args.pretrained.resolve()),
            "pretrained_sha256": file_digest(args.pretrained / "saved_model.pb"),
            "selected": selected,
            "epochs": args.epochs,
            "validation_sha256": file_digest(
                args.notification_validation / "validation.json"
            ),
            "audio_bandwidth": "8000 Hz PCM resampled to 16000 Hz; no additional high-frequency information",
            "pretraining": "YAMNet AudioSet; pretraining-source overlap with public evaluation audio is not audited",
        },
    )

import json
from dataclasses import replace
from importlib.metadata import version

import numpy as np

from .audio import Frontend
from .augmentation import filter_augment, spec_augment
from .data import (
    LABELS,
    check_independent,
    file_digest,
    inventory,
    load_hard_negatives,
    make_examples,
    provenance,
    read_manifests,
    write_json,
)
from .metrics import classification_metrics
from .model import build_model, export_model, float_predictions


def configure_tensorflow(device: str, seed: int):
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise ValueError("No TensorFlow GPU found; use --extra gpu inside nix develop")
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    else:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def export_checkpoint(args):
    metadata = json.loads((args.model / "metadata.json").read_text())
    history = json.loads((args.model / "history.json").read_text())
    row = next(
        (
            row
            for row in history
            if row["epoch"] == args.epoch and "notifications" in row
        ),
        None,
    )
    if row is None:
        raise ValueError("Checkpoint was not evaluated on notification validation")
    tf = configure_tensorflow("cpu", metadata["seed"])
    checkpoint = args.model / "checkpoints" / f"epoch-{args.epoch:03d}.keras"
    model = tf.keras.models.load_model(checkpoint, compile=False)
    config = Frontend(**metadata["frontend"])
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.model / "calibration.npz") as calibration:
        _, info = export_model(
            model,
            calibration["x"],
            calibration["y"],
            calibration["pcm"],
            config,
            args.output,
            65536,
            tuple(metadata["channels"]),
        )
    model.save(args.output / "model.keras")
    metadata |= info | {
        "selected_epoch": args.epoch,
        "checkpoint_selection": "explicit_validation_shortlist",
        "checkpoint_sha256": file_digest(checkpoint),
    }
    metadata["notification_validation"]["selected"] = row["notifications"]
    write_json(args.output / "metadata.json", metadata)


def train(args) -> dict:
    if args.epochs < 1 or args.batch_size < 1 or args.max_model_bytes < 1:
        raise ValueError("Epochs, batch size, and model budget must be positive")
    if not np.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.1:
        raise ValueError("Learning rate must be positive and at most 0.1")
    channels = tuple(args.channels)
    if len(channels) != 3 or any(c < 1 for c in channels):
        raise ValueError("Specify three positive convolution channel counts")
    records = read_manifests(args.manifest)
    config = Frontend(compression=getattr(args, "compression", "log"))
    if config.compression != "log" and getattr(args, "filter_probability", 0.0):
        raise ValueError("FilterAugment currently requires log-mel features")
    selection_interval = getattr(args, "selection_interval", 4)
    if selection_interval < 1:
        raise ValueError("Checkpoint selection interval must be positive")
    inherited_sources = []
    if args.initial_model:
        initial = json.loads((args.initial_model / "metadata.json").read_text())
        inherited_sources = initial["sources"]
        if Frontend(**initial["frontend"]) != config or initial["labels"] != list(
            LABELS
        ):
            raise ValueError("Initial model has incompatible frontend or labels")
        for split in ("val", "test"):
            check_independent(
                [record for record in records if record.split == split], initial, split
            )
    hard_negatives = load_hard_negatives(args.hard_negatives, records)
    teacher_path = getattr(args, "teacher_model", None)
    if teacher_path:
        teacher_metadata = json.loads((teacher_path / "metadata.json").read_text())
        for split in ("val", "test"):
            check_independent(
                [record for record in records if record.split == split],
                teacher_metadata,
                split,
            )
        inherited_sources.extend(teacher_metadata["sources"])
    sources = list(
        {
            (row["path"], row["group"], row["split"], row["sha256"]): row
            for row in inherited_sources + provenance(records)
        }.values()
    )
    notification_validation = getattr(args, "notification_validation", None)
    validator = None
    if notification_validation:
        from .selection import NotificationValidator, notification_key

        validator = NotificationValidator(
            notification_validation, args.decision_binary, config, sources
        )

    def training_examples(seed):
        return make_examples(
            records,
            "train",
            args.examples_per_class,
            seed,
            config,
            args.synthetic_fraction,
            args.background_multiplier,
            hard_negatives,
            args.hard_negative_fraction,
            args.time_shift_seconds,
            soundscape_fraction=args.soundscape_fraction,
            min_snr_db=args.min_snr_db,
        )

    x, y, pcm = training_examples(args.seed + 1_000_000)
    validation_records = [r for r in records if r.split == "val"]
    validation_complete = set(LABELS).issubset({r.label for r in validation_records})
    validation = None
    validation_status = "unvalidated"
    if not validation_complete and any(
        r.label == "background" for r in validation_records
    ):
        validation_records.extend(
            replace(record, split="val")
            for record in records
            if record.split == "train" and record.label != "background"
        )
        validation_status = "known_chimes_heldout_backgrounds"
    elif validation_complete:
        validation_status = "independent_sources_synthetic_mixtures"
    if validation_status != "unvalidated":
        validation = make_examples(
            validation_records,
            "val",
            args.validation_per_class,
            args.seed + 1,
            config,
            soundscape_fraction=args.soundscape_fraction,
            min_snr_db=args.min_snr_db,
        )
    else:
        print(
            "No complete validation split: this run cannot measure generalization.",
            flush=True,
        )
    tf = configure_tensorflow(args.device, args.seed)
    teacher = None
    loss = "sparse_categorical_crossentropy"
    if teacher_path:
        from .distillation import Teacher, distillation_loss

        teacher = Teacher(teacher_path)
        loss = distillation_loss(
            args.distillation_weight, args.distillation_temperature
        )
    args.output.mkdir(parents=True, exist_ok=False)
    if validator:
        (args.output / "checkpoints").mkdir()
        calibration = np.concatenate(
            [np.flatnonzero(y == index)[:100] for index in range(len(LABELS))]
        )
        np.savez_compressed(
            args.output / "calibration.npz",
            x=x[calibration],
            y=y[calibration],
            pcm=pcm[calibration],
        )
    history = []
    best_loss, best_weights, stale, best_epoch = float("inf"), None, 0, None
    best_notification = None
    with tf.device("/GPU:0" if args.device == "gpu" else "/CPU:0"):
        model = build_model(config, channels)
        if args.initial_model:
            previous = tf.keras.models.load_model(
                args.initial_model / "model.keras", compile=False
            )
            model.set_weights(previous.get_weights())
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=loss,
            jit_compile=False,
        )
        rng = np.random.default_rng(args.seed)
        for epoch in range(args.epochs):
            epoch_x, epoch_y, epoch_pcm = training_examples(args.seed + epoch)
            targets = epoch_y
            if teacher:
                targets = np.concatenate(
                    (
                        np.eye(len(LABELS), dtype=np.float32)[epoch_y],
                        teacher.predict(epoch_pcm),
                    ),
                    axis=1,
                )
            filtered_x = filter_augment(
                epoch_x,
                rng,
                getattr(args, "filter_db", 3.0),
                getattr(args, "filter_probability", 0.0),
                config.log_floor,
            )
            masked_x = spec_augment(
                filtered_x,
                rng,
                args.time_mask_frames,
                args.frequency_mask_bins,
                args.mask_probability,
            )
            model.reset_metrics()
            order = rng.permutation(len(epoch_y))
            for start in range(0, len(epoch_y), args.batch_size):
                selected = order[start : start + args.batch_size]
                model.train_on_batch(masked_x[selected], targets[selected])
            prediction = float_predictions(model, epoch_x)
            row = {
                "epoch": epoch + 1,
                "augmentation_seed": args.seed + epoch,
                "loss": float(
                    -np.log(
                        np.maximum(prediction[np.arange(len(epoch_y)), epoch_y], 1e-7)
                    ).mean()
                ),
                "accuracy": float(np.mean(prediction.argmax(axis=1) == epoch_y)),
            }
            if not np.isfinite(row["loss"]):
                raise ValueError("Training loss is not finite")
            if validation is not None:
                vx, vy, _ = validation
                prediction = float_predictions(model, vx)
                row["val_loss"] = float(
                    -np.log(np.maximum(prediction[np.arange(len(vy)), vy], 1e-7)).mean()
                )
                row["val_accuracy"] = float(np.mean(prediction.argmax(axis=1) == vy))
                if validator is None and row["val_loss"] < best_loss:
                    best_epoch = epoch + 1
                    best_loss, best_weights, stale = (
                        row["val_loss"],
                        model.get_weights(),
                        0,
                    )
                elif validator is None:
                    stale += 1
            if validator and (
                (epoch + 1) % selection_interval == 0 or epoch + 1 == args.epochs
            ):
                row["notifications"] = validator.evaluate(model)
                key = notification_key(row["notifications"])
                model.save(args.output / "checkpoints" / f"epoch-{epoch + 1:03d}.keras")
                if best_notification is None or key < best_notification:
                    best_notification = key
                    best_epoch, best_weights = epoch + 1, model.get_weights()
            history.append(row)
            print(json.dumps(row), flush=True)
            if validator is None and validation is not None and stale >= 8:
                break
        if best_weights is not None:
            model.set_weights(best_weights)
        model.save(args.output / "model.keras")
        train_metrics = classification_metrics(y, float_predictions(model, x))
    with tf.device("/CPU:0"):
        runtime, export_info = export_model(
            model, x, y, pcm, config, args.output, args.max_model_bytes, channels
        )
    metrics = {
        "validation_status": validation_status,
        "training": train_metrics,
        "synthetic_validation_float": None,
        "synthetic_validation_int8": None,
    }
    if validation is not None:
        vx, vy, _ = validation
        metrics["synthetic_validation_float"] = classification_metrics(
            vy, float_predictions(model, vx)
        )
        metrics["synthetic_validation_int8"] = classification_metrics(
            vy, runtime.predict(vx)
        )
    metadata = {
        **export_info,
        "labels": list(LABELS),
        "frontend": config.to_dict(),
        "default_threshold": 0.8,
        "seed": args.seed,
        "device": args.device,
        "examples_per_class": args.examples_per_class,
        "positive_sampling": "uniform_session_then_clip",
        "epochs_completed": len(history),
        "selected_epoch": best_epoch or len(history),
        "checkpoint_selection": "validation_notifications_float_shortlist"
        if validator
        else "minimum_validation_cross_entropy"
        if validation is not None
        else "last_epoch",
        "notification_validation": {
            "sha256": file_digest(notification_validation / "validation.json"),
            "decision_binary_sha256": file_digest(args.decision_binary),
            "selection_interval": selection_interval,
            "selected": history[best_epoch - 1]["notifications"],
            "requires_int8_firmware_validation": True,
        }
        if validator
        else None,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "distillation": {
            "teacher_metadata_sha256": file_digest(teacher_path / "metadata.json"),
            "teacher_model_sha256": file_digest(teacher_path / "model.keras"),
            "weight": args.distillation_weight,
            "temperature": args.distillation_temperature,
            "targets": "computed from each augmented PCM waveform before feature masks",
        }
        if teacher
        else None,
        "initial_model_sha256": file_digest(args.initial_model / "model.int8.tflite")
        if args.initial_model
        else None,
        "validation_per_class": args.validation_per_class,
        "model_parameters": model.count_params(),
        "channels": list(channels),
        "synthetic_fraction": args.synthetic_fraction,
        "augmentation": "regenerated_each_epoch",
        "time_shift_seconds": args.time_shift_seconds,
        "training_snr_db": [args.min_snr_db, 25],
        "soundscape_fraction": args.soundscape_fraction,
        "soundscape_snr_definition": "Scaper integrated loudness; legacy mixtures use active-sample RMS",
        "spec_augment": {
            "time_frames": args.time_mask_frames,
            "frequency_bins": args.frequency_mask_bins,
            "probability": args.mask_probability,
        },
        "filter_augment": {
            "max_db": getattr(args, "filter_db", 3.0),
            "probability": getattr(args, "filter_probability", 0.0),
        },
        "training_rms_dbfs": [-45, -6],
        "background_multiplier": args.background_multiplier,
        "hard_negative_fraction": args.hard_negative_fraction,
        "hard_negative_sources": provenance(
            [item.recording for item in hard_negatives]
        ),
        "hard_negative_windows": [
            {"path": str(item.recording.path), "start_sample": item.start_sample}
            for item in hard_negatives
        ],
        "hard_negative_report_sha256": file_digest(args.hard_negatives)
        if args.hard_negatives
        else None,
        "inventory": inventory(records),
        "sources": sources,
        "versions": {
            name: version(name)
            for name in (
                "tensorflow",
                "keras",
                "ai-edge-litert",
                "numpy",
                "librosa",
                "scaper",
            )
        },
        "validation_status": metrics["validation_status"],
    }
    write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "history.json", history)
    write_json(args.output / "metrics.json", metrics)
    print(
        f"Exported {export_info['model_bytes']} byte int8 model to {args.output}",
        flush=True,
    )
    return metadata

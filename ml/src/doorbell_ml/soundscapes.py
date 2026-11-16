from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from .audio import Frontend


class _ScaperNumpy:
    # Python 3.13 needs NumPy 2; Scaper 1.6.5 still uses the removed Inf alias.
    Inf = np.inf

    def __getattr__(self, name):
        return getattr(np, name)


class SoundscapeCorpus:
    def __init__(self, records, config: Frontend):
        self.temporary = TemporaryDirectory(prefix="doorbell-scaper-")
        self.root = Path(self.temporary.name)
        self.config = config
        self.paths = {}
        for index, record in enumerate(records):
            role = "background" if record.label == "background" else "foreground"
            folder = self.root / role / record.label
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{index}{record.path.suffix}"
            path.symlink_to(record.path.resolve())
            self.paths[record] = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.temporary.cleanup()

    def render(self, background, events, rng, *, duration=12.0, augment=False):
        import scaper

        scaper.core.np = _ScaperNumpy()
        source_duration = sf.info(background.path).duration
        scene = scaper.Scaper(
            duration,
            str(self.root / "foreground"),
            str(self.root / "background"),
            protected_labels=["background"] if source_duration < duration else [],
            random_state=int(rng.integers(2**31)),
        )
        scene.sr = self.config.sample_rate
        scene.ref_db = float(rng.uniform(-38, -20))
        scene.add_background(
            label=("const", "background"),
            source_file=("const", str(self.paths[background])),
            source_time=(
                "const",
                float(rng.uniform(0, max(0, source_duration - duration))),
            ),
        )
        for record, onset, snr in events:
            scene.add_event(
                label=("const", record.label),
                source_file=("const", str(self.paths[record])),
                source_time=("const", 0),
                event_time=("const", onset),
                event_duration=("const", sf.info(record.path).duration),
                snr=("const", snr),
                pitch_shift=("uniform", -0.35, 0.35) if augment else None,
                time_stretch=("uniform", 0.96, 1.04) if augment else None,
            )
        audio, _, annotations, _ = scene.generate(fix_clipping=True)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if (
            len(audio) != round(duration * self.config.sample_rate)
            or not np.isfinite(audio).all()
        ):
            raise ValueError("Scaper returned invalid audio")
        labels = [
            {"onset": start, "offset": stop, "label": label}
            for start, stop, label in annotations
        ]
        return audio, labels


def stream_windows(audio, events, config: Frontend, *, hop_samples=2048):
    from .data import LABELS

    if hop_samples < 1:
        raise ValueError("Prediction hop must be positive")
    windows, labels, times = [], [], []
    for start in range(0, len(audio) - config.window_samples + 1, hop_samples):
        end = start + config.window_samples
        overlaps = [
            (
                event,
                max(
                    0,
                    min(end / config.sample_rate, event["offset"])
                    - max(start / config.sample_rate, event["onset"]),
                ),
            )
            for event in events
        ]
        overlaps = [(event, overlap) for event, overlap in overlaps if overlap > 0]
        label = 0
        if overlaps:
            # Shared opening tones cannot yet identify which chime is playing.
            if len(overlaps) != 1 or overlaps[0][1] < 0.75:
                continue
            label = LABELS.index(overlaps[0][0]["label"])
        windows.append(audio[start:end])
        labels.append(label)
        times.append(end / config.sample_rate)
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(times),
    )

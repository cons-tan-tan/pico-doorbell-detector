from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gcd
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly, sosfilt


@dataclass(frozen=True)
class Frontend:
    sample_rate: int = 8000
    window_samples: int = 16000
    frame_samples: int = 256
    hop_samples: int = 128
    mel_bins: int = 20
    low_hz: float = 80.0
    high_hz: float = 3800.0
    log_floor: float = -10.0
    compression: str = "log"
    pcen_alpha: float = 0.98
    pcen_delta: float = 2.0
    pcen_root: float = 0.5
    pcen_smooth: float = 0.04
    pcen_epsilon: float = 1e-10
    pcen_scale: float = 4.0

    @property
    def shape(self) -> tuple[int, int, int]:
        frames = 1 + (self.window_samples - self.frame_samples) // self.hop_samples
        return frames, self.mel_bins, 1

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_FRONTEND = Frontend()


@lru_cache(maxsize=128)
def load_audio(path: Path, sample_rate: int = 8000) -> np.ndarray:
    samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    if source_rate != sample_rate:
        divisor = gcd(source_rate, sample_rate)
        samples = resample_poly(samples, sample_rate // divisor, source_rate // divisor)
    samples = np.asarray(samples, dtype=np.float32)
    samples.setflags(write=False)
    return samples


@lru_cache(maxsize=8)
def coefficients(config: Frontend) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(config.frame_samples + 1)[:-1].astype(np.float32)
    mel = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.frame_samples,
        n_mels=config.mel_bins,
        fmin=config.low_hz,
        fmax=config.high_hz,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    )
    return window, mel


def features(samples: np.ndarray, config: Frontend = DEFAULT_FRONTEND) -> np.ndarray:
    if samples.shape != (config.window_samples,) or not np.isfinite(samples).all():
        raise ValueError(f"Expected {config.window_samples} finite mono PCM samples")
    return feature_sequence(samples, config)


def feature_sequence(
    samples: np.ndarray, config: Frontend = DEFAULT_FRONTEND
) -> np.ndarray:
    if (
        samples.ndim != 1
        or len(samples) < config.frame_samples
        or not np.isfinite(samples).all()
    ):
        raise ValueError("Expected finite mono PCM with at least one FFT frame")
    window, mel = coefficients(config)
    frames = np.lib.stride_tricks.sliding_window_view(samples, config.frame_samples)
    frames = frames[:: config.hop_samples].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.abs(spectrum) ** 2 / config.frame_samples**2
    energy = power @ mel.T
    if config.compression == "pcen":
        smooth = lfilter(
            [config.pcen_smooth], [1, config.pcen_smooth - 1], energy, axis=0
        )
        normalized = energy / (config.pcen_epsilon + smooth) ** config.pcen_alpha
        compressed = (
            normalized + config.pcen_delta
        ) ** config.pcen_root - config.pcen_delta**config.pcen_root
        return np.clip(compressed / config.pcen_scale, 0, 1).astype(np.float32)[
            ..., np.newaxis
        ]
    if config.compression != "log":
        raise ValueError("Unknown feature compression")
    log_energy = np.log10(np.maximum(energy, 10.0**config.log_floor))
    scaled = (
        np.clip(log_energy, config.log_floor, 0) - config.log_floor
    ) / -config.log_floor
    return scaled.astype(np.float32)[..., np.newaxis]


def streaming_features(audio, config):
    pcm = np.pad(audio, (config.window_samples, 0))
    adc = np.clip(np.floor(pcm * 2048 + 2048 + 0.5), 0, 4095)
    sequence = feature_sequence((adc - 2048).astype(np.float32) / 2048, config)
    hop = 2048
    windows = [
        sequence[
            start // config.hop_samples : start // config.hop_samples + config.shape[0]
        ]
        for start in range(hop, len(audio) + 1, hop)
    ]
    return np.stack(windows) if windows else np.empty((0, *config.shape), np.float32)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def crop_background(
    samples: np.ndarray, length: int, rng: np.random.Generator
) -> np.ndarray:
    if len(samples) < length:
        offset = int(rng.integers(length - len(samples) + 1))
        result = np.zeros(length, dtype=np.float32)
        result[offset : offset + len(samples)] = samples
        return result
    start = int(rng.integers(len(samples) - length + 1))
    return samples[start : start + length].copy()


def crop_event(
    samples: np.ndarray, length: int, rng: np.random.Generator
) -> np.ndarray:
    samples = samples - samples.mean()
    frame = 80
    padded = np.pad(samples, (0, (-len(samples)) % frame))
    energy = np.mean(padded.reshape(-1, frame) ** 2, axis=1)
    if energy.max() < 1e-10:
        raise ValueError("A doorbell recording is silent")
    active = np.flatnonzero(energy > energy.max() * 0.01)
    start = max(0, int(active[0]) * frame - 400)
    stop = min(len(samples), (int(active[-1]) + 1) * frame + 400)
    samples = samples[start:stop]
    if len(samples) <= length:
        return crop_background(samples, length, rng)
    starts = np.arange(0, len(samples) - length + 1, frame)
    cumulative = np.concatenate(([0.0], np.cumsum(samples.astype(np.float64) ** 2)))
    energy = cumulative[starts + length] - cumulative[starts]
    # Unrestricted crops can label silence as a doorbell.
    choices = starts[energy >= energy.max() * 0.8]
    start = int(rng.choice(choices))
    return samples[start : start + length].copy()


def augment_event(
    samples: np.ndarray, rng: np.random.Generator, config: Frontend
) -> np.ndarray:
    speed = int(rng.integers(960, 1041))
    samples = resample_poly(samples, 1000, speed)
    if rng.random() < 0.5:
        cutoff = rng.uniform(1800, 3600)
        samples = sosfilt(
            butter(2, cutoff, fs=config.sample_rate, output="sos"), samples
        )
    if rng.random() < 0.5:
        delay = int(rng.uniform(0.025, 0.12) * config.sample_rate)
        echo = np.pad(samples, (delay, 0))[: len(samples)]
        samples = samples + rng.uniform(0.05, 0.25) * echo
    return np.asarray(samples, dtype=np.float32)


def shift_event(
    samples: np.ndarray, max_shift: int, rng: np.random.Generator
) -> np.ndarray:
    if not 0 <= max_shift < len(samples):
        raise ValueError("Time shift must be shorter than the input window")
    shifts = np.unique(np.append(np.arange(-max_shift, max_shift + 1, 80), 0))
    cumulative = np.concatenate(([0.0], np.cumsum(samples.astype(np.float64) ** 2)))
    starts = np.maximum(0, -shifts)
    stops = np.minimum(len(samples), len(samples) - shifts)
    energy = cumulative[stops] - cumulative[starts]
    shift = int(rng.choice(shifts[energy >= 0.7 * cumulative[-1]]))
    result = np.zeros_like(samples)
    count = len(samples) - abs(shift)
    result[max(0, shift) : max(0, shift) + count] = samples[
        max(0, -shift) : max(0, -shift) + count
    ]
    return result


def mix_at_snr(
    foreground: np.ndarray, background: np.ndarray, snr_db: float
) -> np.ndarray:
    active = np.abs(foreground) > np.max(np.abs(foreground)) * 0.05
    if not active.any():
        raise ValueError("Cannot mix a silent positive example")
    noise_rms = rms(background)
    if noise_rms < 1e-10:
        return foreground.copy()
    scale = rms(foreground[active]) / (noise_rms * 10 ** (snr_db / 20))
    return foreground + background * scale


def random_level(
    samples: np.ndarray,
    rng: np.random.Generator,
    low_db: float = -38,
    high_db: float = -15,
) -> np.ndarray:
    level = rms(samples)
    if level < 1e-10:
        return samples.astype(np.float32)
    target = 10 ** (rng.uniform(low_db, high_db) / 20)
    return np.clip(samples * (target / level), -1, 1).astype(np.float32)

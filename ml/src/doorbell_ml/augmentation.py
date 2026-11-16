import numpy as np


def filter_augment(
    inputs: np.ndarray,
    rng: np.random.Generator,
    max_db: float = 3.0,
    probability: float = 0.5,
    log_floor: float = -10.0,
) -> np.ndarray:
    if (
        inputs.ndim != 4
        or inputs.shape[-1] != 1
        or inputs.shape[2] < 4
        or not np.isfinite([max_db, probability, log_floor]).all()
        or not 0 <= max_db <= 12
        or not 0 <= probability <= 1
        or log_floor >= 0
    ):
        raise ValueError("Invalid log-mel shape or FilterAugment limits")
    result = inputs.copy()
    bins = inputs.shape[2]
    for sample in result:
        if rng.random() >= probability:
            continue
        count = int(rng.integers(3, min(6, bins) + 1))
        boundaries = np.sort(rng.choice(np.arange(1, bins - 1), count - 2, False))
        knots = np.concatenate(([0], boundaries, [bins - 1]))
        gains = rng.uniform(-max_db, max_db, count)
        # The frontend stores log10 power scaled by -log_floor, not decibels.
        offsets = np.interp(np.arange(bins), knots, gains) / (10 * -log_floor)
        sample += offsets[np.newaxis, :, np.newaxis]
    return np.clip(result, 0, 1)


def spec_augment(
    inputs: np.ndarray,
    rng: np.random.Generator,
    time_frames: int = 8,
    frequency_bins: int = 1,
    probability: float = 0.5,
) -> np.ndarray:
    if (
        inputs.ndim != 4
        or not 0 <= time_frames < inputs.shape[1]
        or not 0 <= frequency_bins < inputs.shape[2]
        or not 0 <= probability <= 1
    ):
        raise ValueError("Invalid spectrogram shape or SpecAugment mask limits")
    result = inputs.copy()
    for sample in result:
        if rng.random() >= probability:
            continue
        fill = float(sample.mean())
        for axis, limit in ((0, time_frames), (1, frequency_bins)):
            width = int(rng.integers(limit + 1))
            start = int(rng.integers(sample.shape[axis] - width + 1))
            region = [slice(None)] * sample.ndim
            region[axis] = slice(start, start + width)
            sample[tuple(region)] = fill
    return result

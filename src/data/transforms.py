"""Deterministic transforms applied to every spectrogram at load time.

Stochastic *training-time* augmentation (SpecAugment / Mixup / Gaussian
noise, README §8 "Online Augmentation") lives in `src.training.augment`
instead — it needs to see label pairs and only runs during training, so it
does not belong on the `Dataset.__getitem__` path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Compose:
    """Chain several callables, each `array -> array`."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


@dataclass
class PadOrTrim:
    """Force the time axis (last axis) to a fixed number of frames.

    Segmentation already produces fixed-length windows (README §5.3), but
    this guards against off-by-one frame counts from librosa/torchaudio
    STFT edge handling so every batch tensor has an identical shape.
    """

    target_frames: int

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n_frames = x.shape[-1]
        if n_frames == self.target_frames:
            return x
        if n_frames > self.target_frames:
            start = (n_frames - self.target_frames) // 2
            return x[..., start : start + self.target_frames]
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, self.target_frames - n_frames)]
        return np.pad(x, pad_width, mode="constant", constant_values=x.min() if x.size else 0.0)


@dataclass
class ZScoreNormalize:
    """Per-sample z-score normalisation (README §5.5, step 1 of 2)."""

    eps: float = 1e-8

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + self.eps)


@dataclass
class MinMaxScale:
    """Global min-max scaling to [0, 1] (README §5.5, step 2 of 2).

    "Global" here means fixed dataset-level bounds fit once on the training
    set (`fit`), not a per-sample min/max, so validation/test tensors are
    scaled consistently with training.
    """

    data_min: float = -1.0
    data_max: float = 1.0
    eps: float = 1e-8

    def fit(self, samples: list) -> "MinMaxScale":
        self.data_min = float(min(np.min(s) for s in samples))
        self.data_max = float(max(np.max(s) for s in samples))
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        scaled = (x - self.data_min) / (self.data_max - self.data_min + self.eps)
        return np.clip(scaled, 0.0, 1.0)


@dataclass
class ToChannelsFirst:
    """(freq, time) -> (1, freq, time) for a single-channel CNN input."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return x[np.newaxis, ...]
        return x


def build_default_transform(target_frames: int, norm_stats: dict | None = None) -> Compose:
    """Standard inference/eval-time transform pipeline: pad/trim -> z-score -> min-max -> CHW."""
    minmax = MinMaxScale(**norm_stats) if norm_stats else MinMaxScale()
    return Compose(
        [
            PadOrTrim(target_frames),
            ZScoreNormalize(),
            minmax,
            ToChannelsFirst(),
        ]
    )

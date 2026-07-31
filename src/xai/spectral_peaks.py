"""Acoustic sanity check: peak-pick the dominant harmonic series inside a Grad-CAM
highlighted region (README §11.3).

Grounds a Grad-CAM explanation in acoustic theory: propeller/shaft noise
appears as a harmonic series at multiples of the blade-rate frequency
(blade-rate x number of blades = shaft-rate harmonics). If the highlighted
region's peaks form a plausible harmonic series, the explanation is
acoustically consistent with the predicted vessel category; if not, that's
grounds for an operator to question the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class HarmonicSeries:
    fundamental_hz: float
    harmonic_freqs_hz: list[float]
    n_harmonics: int
    mean_spacing_hz: float
    regularity: float  # 0..1, how evenly spaced the peaks are (1 = a perfect harmonic series)


def highlighted_frequency_profile(
    spectrogram: np.ndarray,
    heatmap: np.ndarray,
    heatmap_threshold: float = 0.5,
) -> np.ndarray:
    """Average spectral magnitude across time, restricted to Grad-CAM-highlighted time frames.

    `spectrogram`/`heatmap` are both (freq_bins, time_frames), same shape as
    the Grad-CAM heatmap already resized to the input (README §11.1).
    """
    time_mask = heatmap.max(axis=0) >= heatmap_threshold
    if not time_mask.any():
        time_mask = np.ones(spectrogram.shape[1], dtype=bool)
    return spectrogram[:, time_mask].mean(axis=1)


def freq_bins_to_hz(n_bins: int, sample_rate: int, fmin: float = 0.0) -> np.ndarray:
    """Linear frequency axis for a mel/CQT bin index -> Hz.

    An approximation for mel bins (which are not linearly spaced in Hz);
    good enough for reporting an approximate harmonic spacing to an
    operator, not for precise frequency measurement.
    """
    return np.linspace(fmin, sample_rate / 2, n_bins)


def detect_harmonic_series(
    frequency_profile: np.ndarray,
    freqs_hz: np.ndarray,
    min_peak_prominence: float = 0.1,
    max_harmonics: int = 10,
) -> HarmonicSeries | None:
    """Find spectral peaks and test whether they form an (approximately) evenly-spaced
    harmonic series, i.e. `freq_k ~= k * fundamental`."""
    normalized = (frequency_profile - frequency_profile.min()) / (
        frequency_profile.max() - frequency_profile.min() + 1e-8
    )
    peak_indices, _ = find_peaks(normalized, prominence=min_peak_prominence)
    if len(peak_indices) < 2:
        return None

    peak_freqs = np.sort(freqs_hz[peak_indices])[:max_harmonics]
    spacings = np.diff(peak_freqs)
    mean_spacing = float(np.mean(spacings))
    if mean_spacing <= 0:
        return None

    # Regularity: how close each spacing is to the mean spacing (1 = perfectly even -> harmonic).
    relative_deviation = np.abs(spacings - mean_spacing) / mean_spacing
    regularity = float(np.clip(1.0 - relative_deviation.mean(), 0.0, 1.0))

    return HarmonicSeries(
        fundamental_hz=float(peak_freqs[0]),
        harmonic_freqs_hz=peak_freqs.tolist(),
        n_harmonics=len(peak_freqs),
        mean_spacing_hz=mean_spacing,
        regularity=regularity,
    )


def acoustic_sanity_check(
    spectrogram: np.ndarray,
    heatmap: np.ndarray,
    sample_rate: int,
    fmin: float = 0.0,
) -> dict:
    """End-to-end README §11.3 check: profile -> peaks -> harmonic series summary dict."""
    freqs_hz = freq_bins_to_hz(spectrogram.shape[0], sample_rate, fmin)
    profile = highlighted_frequency_profile(spectrogram, heatmap)
    series = detect_harmonic_series(profile, freqs_hz)

    if series is None:
        return {"harmonic_series_detected": False}

    return {
        "harmonic_series_detected": True,
        "fundamental_hz": round(series.fundamental_hz, 1),
        "harmonic_freqs_hz": [round(f, 1) for f in series.harmonic_freqs_hz],
        "n_harmonics": series.n_harmonics,
        "mean_spacing_hz": round(series.mean_spacing_hz, 1),
        "regularity": round(series.regularity, 3),
    }

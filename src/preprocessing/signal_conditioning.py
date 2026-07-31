"""Signal conditioning: downmix, resample, band-pass filter (README §5.1-5.2)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

TARGET_SAMPLE_RATE = 16000
BANDPASS_LOW_HZ = 10.0
BANDPASS_HIGH_HZ = 8000.0
BANDPASS_ORDER = 4


def downmix_to_mono(waveform: np.ndarray) -> np.ndarray:
    """Collapse a (channels, samples) or (samples, channels) array to mono by averaging."""
    if waveform.ndim == 1:
        return waveform
    # Assume the smaller dimension is the channel axis.
    channel_axis = 0 if waveform.shape[0] < waveform.shape[1] else 1
    return waveform.mean(axis=channel_axis).astype(waveform.dtype)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample mono `waveform` from `orig_sr` to `target_sr` using librosa's polyphase resampler."""
    if orig_sr == target_sr:
        return waveform
    import librosa

    return librosa.resample(waveform.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def bandpass_filter(
    waveform: np.ndarray,
    sample_rate: int,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """4th-order Butterworth band-pass, zero-phase (`sosfiltfilt`) to avoid group delay.

    Removes DC drift / hydrophone self-noise below `low_hz` and anti-aliases
    above `high_hz`, matching README §5.2.
    """
    nyquist = sample_rate / 2.0
    high_hz = min(high_hz, nyquist * 0.99)
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, waveform).astype(np.float32)


def condition_signal(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Full conditioning chain: downmix -> resample -> band-pass filter."""
    mono = downmix_to_mono(waveform)
    resampled = resample(mono, orig_sr, target_sr)
    filtered = bandpass_filter(resampled, target_sr)
    return filtered, target_sr

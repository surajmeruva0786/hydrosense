"""Time-frequency representations benchmarked in README §5.4.

| Representation | Parameters                         | Output shape |
|-----------------|------------------------------------|--------------|
| mel             | n_fft=2048, hop=512, n_mels=128    | (128, 313)   |
| log_mel         | mel + log(1 + x)                   | (128, 313)   |
| cqt             | bins/octave=24, fmin=10 Hz         | (256, 313)   |
| mfcc            | n_mfcc=40 + delta + delta-delta    | (120, 313)   |

Shapes above assume a 10 s window at 16 kHz (README §5.3); other segment
lengths scale the time axis proportionally.
"""

from __future__ import annotations

import numpy as np

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
CQT_BINS_PER_OCTAVE = 24
CQT_FMIN = 10.0
CQT_N_BINS = 256
N_MFCC = 40

REPRESENTATIONS = ("mel", "log_mel", "cqt", "mfcc", "waveform")


def mel_spectrogram(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return mel.astype(np.float32)


def log_mel_spectrogram(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    mel = mel_spectrogram(waveform, sample_rate)
    return np.log1p(mel).astype(np.float32)


def cqt_spectrogram(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    n_octaves = CQT_N_BINS / CQT_BINS_PER_OCTAVE
    max_cqt_freq = CQT_FMIN * (2**n_octaves)
    if max_cqt_freq > sample_rate / 2:
        n_octaves = np.log2((sample_rate / 2) / CQT_FMIN)
        n_bins = int(n_octaves * CQT_BINS_PER_OCTAVE)
    else:
        n_bins = CQT_N_BINS

    cqt = librosa.cqt(
        y=waveform,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        fmin=CQT_FMIN,
        n_bins=n_bins,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
    )
    magnitude = np.abs(cqt).astype(np.float32)
    if magnitude.shape[0] < CQT_N_BINS:
        pad = CQT_N_BINS - magnitude.shape[0]
        magnitude = np.pad(magnitude, ((0, pad), (0, 0)), mode="constant")
    return magnitude


def mfcc_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta, delta2], axis=0).astype(np.float32)


def raw_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG001
    """Passthrough representation for HydroSense-TL, whose YAMNet backbone (README §7.3)
    computes its own internal log-mel features directly from the raw waveform."""
    return waveform.astype(np.float32)


_DISPATCH = {
    "mel": mel_spectrogram,
    "log_mel": log_mel_spectrogram,
    "cqt": cqt_spectrogram,
    "mfcc": mfcc_features,
    "waveform": raw_waveform,
}


def compute_representation(
    waveform: np.ndarray, sample_rate: int, representation: str
) -> np.ndarray:
    """Dispatch to one of the functions above by name (`configs/*.yaml: representation`)."""
    if representation not in _DISPATCH:
        raise ValueError(
            f"Unknown representation '{representation}'. Expected one of {REPRESENTATIONS}."
        )
    return _DISPATCH[representation](waveform, sample_rate)


def expected_num_frames(num_samples: int, hop_length: int = HOP_LENGTH) -> int:
    """librosa's STFT frame count for a signal of `num_samples`, `center=True` (the default)."""
    return 1 + num_samples // hop_length

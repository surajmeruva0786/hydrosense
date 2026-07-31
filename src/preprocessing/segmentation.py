"""Fixed-length windowing with overlap and silence removal (README §5.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SEGMENT_LENGTH_S = 10.0
DEFAULT_OVERLAP = 0.5
DEFAULT_SILENCE_RMS_THRESHOLD = 1e-4


@dataclass
class Segment:
    waveform: np.ndarray
    start_sample: int
    end_sample: int


def segment_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    segment_length_s: float = DEFAULT_SEGMENT_LENGTH_S,
    overlap: float = DEFAULT_OVERLAP,
) -> list[Segment]:
    """Split `waveform` into fixed-length windows with the given fractional overlap.

    The final partial window (shorter than `segment_length_s`) is dropped
    rather than zero-padded, so every emitted segment carries an equal
    amount of real signal.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    window_samples = int(round(segment_length_s * sample_rate))
    hop_samples = max(1, int(round(window_samples * (1.0 - overlap))))

    segments = []
    start = 0
    while start + window_samples <= len(waveform):
        end = start + window_samples
        segments.append(Segment(waveform[start:end], start, end))
        start += hop_samples

    return segments


def is_silent(
    segment: np.ndarray,
    rms_threshold: float = DEFAULT_SILENCE_RMS_THRESHOLD,
) -> bool:
    """RMS-energy silence gate — discards near-empty frames before training (README §5.2)."""
    rms = float(np.sqrt(np.mean(np.square(segment))))
    return rms < rms_threshold


def segment_and_filter_silence(
    waveform: np.ndarray,
    sample_rate: int,
    segment_length_s: float = DEFAULT_SEGMENT_LENGTH_S,
    overlap: float = DEFAULT_OVERLAP,
    rms_threshold: float = DEFAULT_SILENCE_RMS_THRESHOLD,
    keep_silence_for_class: bool = False,
) -> list[Segment]:
    """`segment_waveform` followed by the silence gate.

    `keep_silence_for_class=True` should be passed for category E
    (ambient/no-vessel, README §6) recordings, where near-silence *is* the
    signal of interest rather than noise to discard.
    """
    segments = segment_waveform(waveform, sample_rate, segment_length_s, overlap)
    if keep_silence_for_class:
        return segments
    return [s for s in segments if not is_silent(s.waveform, rms_threshold)]

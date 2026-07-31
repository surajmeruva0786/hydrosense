from __future__ import annotations

import numpy as np

from src.preprocessing.representations import (
    compute_representation,
    expected_num_frames,
    mel_spectrogram,
)
from src.preprocessing.segmentation import is_silent, segment_and_filter_silence, segment_waveform
from src.preprocessing.signal_conditioning import bandpass_filter, condition_signal, downmix_to_mono, resample


def test_downmix_to_mono_averages_channels():
    stereo = np.stack([np.ones(100), np.zeros(100)])  # (2, 100)
    mono = downmix_to_mono(stereo)
    assert mono.shape == (100,)
    np.testing.assert_allclose(mono, 0.5)


def test_resample_changes_length_proportionally():
    waveform = np.random.rand(16000).astype(np.float32)
    resampled = resample(waveform, orig_sr=16000, target_sr=8000)
    assert abs(len(resampled) - 8000) < 10


def test_bandpass_filter_preserves_length():
    waveform = np.random.rand(16000).astype(np.float32)
    filtered = bandpass_filter(waveform, sample_rate=16000)
    assert filtered.shape == waveform.shape


def test_condition_signal_end_to_end(synthetic_waveform, sample_rate):
    conditioned, sr = condition_signal(synthetic_waveform, sample_rate, target_sr=16000)
    assert sr == 16000
    assert conditioned.ndim == 1
    assert len(conditioned) == len(synthetic_waveform)


def test_segment_waveform_overlap_and_count():
    waveform = np.random.rand(16000 * 5).astype(np.float32)  # 5s
    segments = segment_waveform(waveform, sample_rate=16000, segment_length_s=1.0, overlap=0.5)
    assert len(segments) == 9  # hop=0.5s over 5s minus final partial window
    for seg in segments:
        assert len(seg.waveform) == 16000


def test_is_silent_detects_near_zero_energy():
    silent = np.zeros(1000)
    loud = np.ones(1000)
    assert is_silent(silent)
    assert not is_silent(loud)


def test_segment_and_filter_silence_drops_quiet_segments():
    loud = np.ones(16000)
    silent = np.zeros(16000)
    waveform = np.concatenate([loud, silent, loud])
    segments = segment_and_filter_silence(waveform, sample_rate=16000, segment_length_s=1.0, overlap=0.0)
    assert all(not is_silent(s.waveform) for s in segments)


def test_mel_spectrogram_shape_matches_readme_table():
    waveform = np.random.rand(16000 * 10).astype(np.float32)  # 10s @ 16kHz
    mel = mel_spectrogram(waveform, sample_rate=16000)
    assert mel.shape[0] == 128
    assert mel.shape[1] == expected_num_frames(len(waveform))


def test_compute_representation_dispatch_all_known_types():
    waveform = np.random.rand(16000 * 2).astype(np.float32)
    for rep_name in ["mel", "log_mel", "cqt", "mfcc", "waveform"]:
        out = compute_representation(waveform, 16000, rep_name)
        assert out is not None
        assert out.size > 0

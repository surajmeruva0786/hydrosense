from __future__ import annotations

import numpy as np

from src.data.transforms import MinMaxScale, PadOrTrim, ToChannelsFirst, ZScoreNormalize, build_default_transform


def test_pad_or_trim_pads_short_and_trims_long():
    short = np.random.rand(10, 5)
    padded = PadOrTrim(target_frames=8)(short)
    assert padded.shape == (10, 8)

    long = np.random.rand(10, 20)
    trimmed = PadOrTrim(target_frames=8)(long)
    assert trimmed.shape == (10, 8)

    exact = np.random.rand(10, 8)
    assert PadOrTrim(target_frames=8)(exact).shape == (10, 8)


def test_zscore_normalize_zero_mean_unit_std():
    x = np.random.rand(20, 20) * 100 + 50
    z = ZScoreNormalize()(x)
    assert abs(z.mean()) < 1e-5
    assert abs(z.std() - 1.0) < 1e-3


def test_minmax_scale_bounds():
    x = np.array([[-5.0, 0.0, 5.0]])
    scaler = MinMaxScale(data_min=-5.0, data_max=5.0)
    scaled = scaler(x)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    np.testing.assert_allclose(scaled, [[0.0, 0.5, 1.0]], atol=1e-6)


def test_to_channels_first_adds_axis():
    x = np.random.rand(128, 313)
    out = ToChannelsFirst()(x)
    assert out.shape == (1, 128, 313)


def test_build_default_transform_pipeline_shape():
    transform = build_default_transform(target_frames=313)
    x = np.random.rand(128, 300)
    out = transform(x)
    assert out.shape == (1, 128, 313)
    assert out.min() >= 0.0 and out.max() <= 1.0

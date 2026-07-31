from __future__ import annotations

import torch

from src.training.augment import AugmentConfig, gaussian_noise, mixup, spec_augment


def test_spec_augment_preserves_shape():
    cfg = AugmentConfig.from_dict({"specaugment": {"freq_masks": 2, "freq_mask_width": 10, "time_masks": 2, "time_mask_width": 20}})
    x = torch.rand(4, 1, 40, 60)
    out = spec_augment(x, cfg.specaugment, prob=1.0)
    assert out.shape == x.shape


def test_spec_augment_prob_zero_is_noop():
    x = torch.rand(4, 1, 40, 60)
    cfg = AugmentConfig.from_dict({})
    out = spec_augment(x, cfg.specaugment, prob=0.0)
    torch.testing.assert_close(out, x)


def test_gaussian_noise_prob_zero_is_noop():
    x = torch.rand(4, 1, 20, 20)
    out = gaussian_noise(x, std=0.1, prob=0.0)
    torch.testing.assert_close(out, x)


def test_gaussian_noise_changes_values_when_applied():
    torch.manual_seed(0)
    x = torch.rand(4, 1, 20, 20)
    out = gaussian_noise(x, std=0.5, prob=1.0)
    assert not torch.allclose(out, x)
    assert out.shape == x.shape


def test_mixup_disabled_returns_original_with_lam_one():
    x = torch.rand(4, 1, 10, 10)
    y = torch.tensor([0, 1, 2, 3])
    mixed, y_a, y_b, lam = mixup(x, y, alpha=0.0, prob=1.0)
    assert lam == 1.0
    torch.testing.assert_close(mixed, x)
    torch.testing.assert_close(y_a, y)
    torch.testing.assert_close(y_b, y)


def test_mixup_enabled_interpolates_batch():
    torch.manual_seed(0)
    x = torch.rand(8, 1, 10, 10)
    y = torch.arange(8)
    mixed, y_a, y_b, lam = mixup(x, y, alpha=0.2, prob=1.0)
    assert 0.0 <= lam <= 1.0
    assert mixed.shape == x.shape
    torch.testing.assert_close(y_a, y)

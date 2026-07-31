"""Training-time stochastic augmentation (README §8 "Online Augmentation").

Each of the three techniques below is applied independently with
probability 0.5 per sample/batch, matching the README's training procedure:

- **SpecAugment** — 2 frequency masks (<=20 bands), 2 time masks (<=40 frames)
- **Mixup** (alpha=0.2) — convex combination of spectrogram pairs and labels
- **Gaussian noise** — sigma=0.005 added to the normalised spectrogram

These operate on batched `torch.Tensor`s of shape (B, C, F, T) inside the
training loop, unlike `src.data.transforms`, which is deterministic and
applies at `Dataset.__getitem__` time to both train and eval data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SpecAugmentConfig:
    freq_masks: int = 2
    freq_mask_width: int = 20
    time_masks: int = 2
    time_mask_width: int = 40


@dataclass
class AugmentConfig:
    prob: float = 0.5
    specaugment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    mixup_alpha: float = 0.2
    gaussian_noise_std: float = 0.005

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentConfig":
        sa = d.get("specaugment", {})
        return cls(
            prob=d.get("prob", 0.5),
            specaugment=SpecAugmentConfig(
                freq_masks=sa.get("freq_masks", 2),
                freq_mask_width=sa.get("freq_mask_width", 20),
                time_masks=sa.get("time_masks", 2),
                time_mask_width=sa.get("time_mask_width", 40),
            ),
            mixup_alpha=d.get("mixup_alpha", 0.2),
            gaussian_noise_std=d.get("gaussian_noise_std", 0.005),
        )


def spec_augment(batch: torch.Tensor, cfg: SpecAugmentConfig, prob: float = 0.5) -> torch.Tensor:
    """Apply frequency/time masking independently to each sample in the batch with probability `prob`."""
    batch = batch.clone()
    n_freq, n_time = batch.shape[-2], batch.shape[-1]
    fill_value = batch.mean()

    for b in range(batch.shape[0]):
        if torch.rand(1).item() >= prob:
            continue

        for _ in range(cfg.freq_masks):
            width = int(torch.randint(0, min(cfg.freq_mask_width, n_freq) + 1, (1,)).item())
            if width == 0:
                continue
            start = int(torch.randint(0, max(1, n_freq - width + 1), (1,)).item())
            batch[b, ..., start : start + width, :] = fill_value

        for _ in range(cfg.time_masks):
            width = int(torch.randint(0, min(cfg.time_mask_width, n_time) + 1, (1,)).item())
            if width == 0:
                continue
            start = int(torch.randint(0, max(1, n_time - width + 1), (1,)).item())
            batch[b, ..., :, start : start + width] = fill_value

    return batch


def gaussian_noise(batch: torch.Tensor, std: float, prob: float = 0.5) -> torch.Tensor:
    """Additive Gaussian noise per-sample with probability `prob`."""
    if std <= 0:
        return batch
    mask = (torch.rand(batch.shape[0], device=batch.device) < prob).view(-1, *([1] * (batch.dim() - 1)))
    noise = torch.randn_like(batch) * std
    return batch + noise * mask


def mixup(batch: torch.Tensor, labels: torch.Tensor, alpha: float, prob: float = 0.5):
    """Batch-level Mixup (Zhang et al., 2018). Returns (mixed_x, y_a, y_b, lam).

    When skipped (with probability `1 - prob`, or if `alpha <= 0`), returns
    the original batch with `lam=1.0` so `mixup_loss` reduces to the plain
    per-sample loss.
    """
    if alpha <= 0 or torch.rand(1).item() >= prob:
        return batch, labels, labels, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(batch.shape[0], device=batch.device)
    mixed = lam * batch + (1.0 - lam) * batch[perm]
    return mixed, labels, labels[perm], lam


def apply_training_augmentation(batch: torch.Tensor, labels: torch.Tensor, cfg: AugmentConfig):
    """Full online augmentation chain: SpecAugment -> Gaussian noise -> Mixup.

    Returns (augmented_x, y_a, y_b, lam) — pass straight to
    `src.training.losses.mixup_loss`.
    """
    x = spec_augment(batch, cfg.specaugment, prob=cfg.prob)
    x = gaussian_noise(x, cfg.gaussian_noise_std, prob=cfg.prob)
    x, y_a, y_b, lam = mixup(x, labels, cfg.mixup_alpha, prob=cfg.prob)
    return x, y_a, y_b, lam

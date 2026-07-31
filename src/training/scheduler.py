"""Cosine-annealing LR schedule with linear warmup (README §8: "5-epoch warmup")."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warmup for `warmup_epochs`, then cosine decay to `min_lr_ratio * base_lr`.

    Stepped once per epoch (not per batch) to match the README's epoch-level
    schedule description.
    """

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

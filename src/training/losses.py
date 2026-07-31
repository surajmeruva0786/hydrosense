"""Loss construction (README §8: "class-weighted categorical cross-entropy")."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def build_loss(class_weights: np.ndarray | None = None, device: str = "cpu") -> nn.Module:
    """`nn.CrossEntropyLoss`, optionally class-weighted to counter ShipsEar imbalance (README §6)."""
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


def mixup_loss(criterion: nn.Module, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Convex combination of the loss against each of a Mixup pair's labels (README §8 Mixup)."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)

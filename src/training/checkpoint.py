"""Checkpointing and early stopping (README §8: "early stopping, patience 15")."""

from __future__ import annotations

from pathlib import Path

import torch


class EarlyStopping:
    """Stops training when the monitored metric has not improved for `patience` epochs."""

    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float) -> bool:
        """Update state with the latest epoch's score. Returns True iff it's a new best."""
        is_best = self._is_improvement(score)
        if is_best:
            self.best_score = score
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            self.should_stop = self.num_bad_epochs >= self.patience
        return is_best


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    metrics: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, map_location: str = "cpu") -> dict:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state

"""Model factory: `config["model"]` (README §14 `--model`) -> constructed model instance."""

from __future__ import annotations

from typing import Callable

_BUILDERS: dict[str, Callable[[dict], object]] = {}


def register(name: str):
    def _decorator(fn: Callable[[dict], object]):
        _BUILDERS[name] = fn
        return fn

    return _decorator


def build_model(config: dict):
    """Construct the model named by `config["model"]` (one of `hydrosense_base`,
    `hydrosense_se`, `hydrosense_tl`)."""
    model_name = config["model"]
    if model_name not in _BUILDERS:
        raise ValueError(f"Unknown model '{model_name}'. Registered: {sorted(_BUILDERS)}")
    return _BUILDERS[model_name](config)


def _register_builtins() -> None:
    from src.models.hydrosense_base import build_hydrosense_base
    from src.models.hydrosense_se import build_hydrosense_se
    from src.models.hydrosense_tl import build_hydrosense_tl

    _BUILDERS.setdefault("hydrosense_base", build_hydrosense_base)
    _BUILDERS.setdefault("hydrosense_se", build_hydrosense_se)
    _BUILDERS.setdefault("hydrosense_tl", build_hydrosense_tl)


_register_builtins()


def is_torch_model(model) -> bool:
    """True for HydroSense-Base/SE (torch.nn.Module); False for HydroSense-TL (tf.keras.Model)."""
    try:
        import torch

        return isinstance(model, torch.nn.Module)
    except ImportError:
        return False

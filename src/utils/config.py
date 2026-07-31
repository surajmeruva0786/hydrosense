"""YAML experiment configuration loading.

Every training / evaluation run is driven by one of the YAML files in
`configs/`. Loading through this module (rather than raw `yaml.safe_load`)
guarantees defaults are filled in and the file that produced a run is
always recoverable from the run directory.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "hydrosense_base",
    "representation": "mel",
    "num_classes": 5,
    "class_names": ["A", "B", "C", "D", "E"],
    "sample_rate": 16000,
    "segment_length": 10.0,
    "seed": 42,
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 5,
        "early_stopping_patience": 15,
        "mixed_precision": True,
        "grad_clip_norm": 1.0,
        "use_synthetic": False,
    },
    "augmentation": {
        "prob": 0.5,
        "specaugment": {"freq_masks": 2, "freq_mask_width": 20, "time_masks": 2, "time_mask_width": 40},
        "mixup_alpha": 0.2,
        "gaussian_noise_std": 0.005,
    },
    "folds": 5,
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, merged on top of `DEFAULT_CONFIG`.

    Unknown top-level keys are preserved as-is so configs can carry
    model-specific extras (e.g. the TL backbone name) without touching
    this module.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = _deep_merge(DEFAULT_CONFIG, raw)
    config["_source_path"] = str(path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Persist an (effective) config next to a run's checkpoints for provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=False)

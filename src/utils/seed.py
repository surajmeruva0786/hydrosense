"""Deterministic seeding across numpy / torch / python for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic_torch: bool = True) -> None:
    """Seed all RNGs used across the pipeline (python, numpy, torch, tensorflow if present).

    Args:
        seed: Seed value. Defaults to the project-wide seed (42) used for every
            reported result in the README.
        deterministic_torch: If True, forces CuDNN / torch algorithm determinism.
            Disable only for throughput-sensitive, non-reported runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

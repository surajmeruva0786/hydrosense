"""SHAP DeepExplainer attribution over mel bands (README §11.2).

Computes per-frequency-band Shapley values with `shap.DeepExplainer`,
averaged across all test samples of a class, giving a global,
class-conditional view of which mel bands carry the strongest evidence for
each vessel category — complementary to Grad-CAM's per-prediction, local
explanation (README §11.1).
"""

from __future__ import annotations

import numpy as np
import torch


def build_background(dataset, n_background: int = 50, seed: int = 42) -> torch.Tensor:
    """Sample a small random background set required by `shap.DeepExplainer`
    (its baseline for the expected-value term of the Shapley decomposition)."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_background, len(dataset)), replace=False)
    samples = [torch.as_tensor(dataset[i][0]) for i in indices]
    return torch.stack(samples).float()


def compute_shap_values(model: torch.nn.Module, background: torch.Tensor, samples: torch.Tensor):
    """Run `shap.DeepExplainer` and return per-class SHAP value arrays, each
    shaped like `samples` (n_samples, C, F, T)."""
    import shap

    model.eval()
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(samples)
    return shap_values


def per_band_attribution(shap_values: np.ndarray, class_idx: int) -> np.ndarray:
    """Average |SHAP| over samples and time for one class -> a (freq_bins,) importance profile.

    `shap_values` is the DeepExplainer output for a single class, shape
    (n_samples, 1, F, T) for our single-channel spectrogram inputs.
    """
    values = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    values = np.asarray(values)
    if values.ndim == 5:  # (n_samples, C, F, T, n_classes) — some shap versions stack classes last
        values = values[..., class_idx]
    return np.abs(values).mean(axis=(0, 1, 3))  # -> (F,)


def class_conditional_band_importance(
    model: torch.nn.Module,
    background: torch.Tensor,
    samples_by_class: dict[int, torch.Tensor],
) -> dict[int, np.ndarray]:
    """README §11.2: per-frequency-band Shapley values averaged across all test
    samples of each class. Returns {class_idx: (freq_bins,) importance array}."""
    result = {}
    for class_idx, samples in samples_by_class.items():
        shap_values = compute_shap_values(model, background, samples)
        result[class_idx] = per_band_attribution(shap_values, class_idx)
    return result

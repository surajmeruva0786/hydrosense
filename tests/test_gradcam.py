from __future__ import annotations

import numpy as np
import torch

from src.models.hydrosense_base import HydroSenseBase
from src.xai.gradcam import GradCAM, overlay_heatmap
from src.xai.spectral_peaks import acoustic_sanity_check


def test_gradcam_output_shape_matches_input_spatial_dims():
    model = HydroSenseBase(num_classes=5)
    cam = GradCAM(model, model.target_layer)
    x = torch.randn(1, 1, 64, 96, requires_grad=True)

    heatmap, class_idx = cam(x)
    cam.remove()

    assert heatmap.shape == (64, 96)
    assert 0 <= class_idx < 5
    assert heatmap.min() >= 0.0 - 1e-6
    assert heatmap.max() <= 1.0 + 1e-6


def test_gradcam_respects_explicit_class_idx():
    model = HydroSenseBase(num_classes=5)
    cam = GradCAM(model, model.target_layer)
    x = torch.randn(1, 1, 32, 32, requires_grad=True)

    _, class_idx = cam(x, class_idx=3)
    cam.remove()
    assert class_idx == 3


def test_overlay_heatmap_returns_valid_rgb():
    spec = np.random.rand(32, 32)
    heatmap = np.random.rand(32, 32)
    overlay = overlay_heatmap(spec, heatmap)

    assert overlay.shape == (32, 32, 3)
    assert overlay.min() >= 0.0 and overlay.max() <= 1.0


def test_acoustic_sanity_check_runs_without_error():
    spec = np.random.rand(128, 64)
    heatmap = np.random.rand(128, 64)
    result = acoustic_sanity_check(spec, heatmap, sample_rate=16000)
    assert "harmonic_series_detected" in result

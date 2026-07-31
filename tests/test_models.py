from __future__ import annotations

import pytest
import torch

from src.models.hydrosense_base import HydroSenseBase
from src.models.hydrosense_se import HydroSenseSE
from src.models.registry import build_model, is_torch_model


@pytest.mark.parametrize("model_cls", [HydroSenseBase, HydroSenseSE])
def test_model_forward_shape_and_gradient_flow(model_cls):
    model = model_cls(num_classes=5)
    x = torch.randn(2, 1, 128, 313, requires_grad=True)
    logits = model(x)

    assert logits.shape == (2, 5)

    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.parametrize("model_cls", [HydroSenseBase, HydroSenseSE])
def test_model_target_layer_is_last_conv(model_cls):
    model = model_cls(num_classes=5)
    assert model.target_layer is model.conv_blocks[-1].conv


def test_registry_builds_base_and_se():
    for model_name in ["hydrosense_base", "hydrosense_se"]:
        config = {"model": model_name, "num_classes": 5, "model_params": {}}
        model = build_model(config)
        assert is_torch_model(model)
        x = torch.randn(1, 1, 64, 64)
        assert model(x).shape == (1, 5)


def test_registry_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model({"model": "not_a_real_model"})

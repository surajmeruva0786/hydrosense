"""HydroSense-Base: custom 4-block 2D CNN for spectrogram classification (README §7.1).

    Input: (1, 128, 313)
    Conv2D(32,3x3)+BN+ReLU+MaxPool(2,2)   -> (32, 64, 156)
    Conv2D(64,3x3)+BN+ReLU+MaxPool(2,2)   -> (64, 32, 78)
    Conv2D(128,3x3)+BN+ReLU+MaxPool(2,2)  -> (128, 16, 39)
    Conv2D(256,3x3)+BN+ReLU+MaxPool(2,2)  -> (256, 8, 19)
    GlobalAveragePool2D                    -> (256,)
    Dropout(0.4)
    Dense(128)+ReLU
    Dropout(0.3)
    Dense(5)+Softmax

~480k parameters. Returns logits (softmax is applied by the loss / at
inference time), and exposes `features` for Grad-CAM (README §11.1) to hook
into the last convolutional block.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import ConvBlock


class HydroSenseBase(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        channels: list[int] | None = None,
        dropout1: float = 0.4,
        dropout2: float = 0.3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        channels = channels or [32, 64, 128, 256]

        blocks = []
        c_in = in_channels
        for c_out in channels:
            blocks.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.conv_blocks = nn.ModuleList(blocks)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(channels[-1], hidden_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self._last_conv_features: torch.Tensor | None = None

    @property
    def target_layer(self) -> nn.Module:
        """Last convolutional block — the standard Grad-CAM hook point (README §11.1)."""
        return self.conv_blocks[-1].conv

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        self._last_conv_features = x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


def build_hydrosense_base(config: dict) -> HydroSenseBase:
    params = config.get("model_params", {})
    return HydroSenseBase(
        num_classes=config.get("num_classes", 5),
        in_channels=params.get("in_channels", 1),
        channels=params.get("channels"),
        dropout1=params.get("dropout1", 0.4),
        dropout2=params.get("dropout2", 0.3),
        hidden_dim=params.get("hidden_dim", 128),
    )

"""HydroSense-SE: HydroSense-Base with a Squeeze-and-Excitation block per conv stage (README §7.2).

Identical topology to `HydroSenseBase` otherwise; the SE gate lets the
network re-weight channels by global context, which the README reports
improves recall on the under-represented ShipsEar categories (D, E).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import SEConvBlock


class HydroSenseSE(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        channels: list[int] | None = None,
        se_reduction: int = 16,
        dropout1: float = 0.4,
        dropout2: float = 0.3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        channels = channels or [32, 64, 128, 256]

        blocks = []
        c_in = in_channels
        for c_out in channels:
            blocks.append(SEConvBlock(c_in, c_out, se_reduction=se_reduction))
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


def build_hydrosense_se(config: dict) -> HydroSenseSE:
    params = config.get("model_params", {})
    return HydroSenseSE(
        num_classes=config.get("num_classes", 5),
        in_channels=params.get("in_channels", 1),
        channels=params.get("channels"),
        se_reduction=params.get("se_reduction", 16),
        dropout1=params.get("dropout1", 0.4),
        dropout2=params.get("dropout2", 0.3),
        hidden_dim=params.get("hidden_dim", 128),
    )

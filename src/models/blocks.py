"""Reusable building blocks shared by the HydroSense model variants."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool, the repeated unit in HydroSense-Base (README §7.1)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-attention block (Hu et al. 2018, README §7.2).

    Squeezes each channel to a scalar via global average pooling, learns a
    per-channel gate through a small bottleneck MLP, and rescales the
    original feature map — letting the network re-weight channels by global
    context rather than only local receptive fields.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        gate = self.avg_pool(x).view(b, c)
        gate = self.fc(gate).view(b, c, 1, 1)
        return x * gate


class SEConvBlock(nn.Module):
    """`ConvBlock` with an SE block inserted before pooling (README §7.2)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool: int = 2, se_reduction: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, reduction=se_reduction)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = self.se(x)
        return self.pool(x)

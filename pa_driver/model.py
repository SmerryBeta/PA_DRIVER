from __future__ import annotations

import torch
from torch import nn


class DrivingCNN(nn.Module):
    def __init__(self, in_channels: int = 4, action_dim: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.features(x))
        steering = torch.tanh(raw[:, 0:1])
        pedals = torch.sigmoid(raw[:, 1:3])
        return torch.cat([steering, pedals], dim=1)

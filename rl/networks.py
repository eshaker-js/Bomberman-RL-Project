"""A small CNN feature extractor for the 11x11 grid observation (v0).

Stable-Baselines3's default image network (``NatureCNN``) uses 8x8 and 4x4
strided convolutions and needs inputs of at least 36x36, so it cannot take
our ``(20, 11, 11)`` observation. This network keeps the 11x11 resolution:
three 3x3 convolutions with padding 1 (channel sizes from
``PPOConfig.conv_channels``), then one linear layer of ``features_dim``.

Written following the "Custom Feature Extractor" pattern of the SB3
documentation (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html).
SB3: A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning
Implementations", JMLR 22(268), 2021.
"""

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SmallGridCNN(BaseFeaturesExtractor):
    """``(C, H, W)`` grid -> ``features_dim`` features, without downsampling."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256,
                 conv_channels: tuple = (32, 64, 64)) -> None:
        super().__init__(observation_space, features_dim)
        channels, height, width = observation_space.shape
        layers = []
        for out_channels in conv_channels:
            layers += [nn.Conv2d(channels, out_channels, kernel_size=3, padding=1), nn.ReLU()]
            channels = out_channels
        self.cnn = nn.Sequential(*layers, nn.Flatten())
        self.linear = nn.Sequential(nn.Linear(channels * height * width, features_dim),
                                    nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

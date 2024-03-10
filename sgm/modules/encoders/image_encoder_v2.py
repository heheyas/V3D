"""
UNet Network in PyTorch, modified from https://github.com/milesial/Pytorch-UNet
with architecture referenced from https://keras.io/examples/vision/depth_estimation
for monocular depth estimation from RGB images, i.e. one output channel.
"""

import torch
from torch import nn


class UNet(nn.Module):
    """
    The overall UNet architecture.
    """

    def __init__(self):
        super().__init__()

        self.downscale_blocks = nn.ModuleList(
            [
                DownBlock(16, 32),
                DownBlock(32, 64),
                DownBlock(64, 128),
                DownBlock(128, 256),
            ]
        )
        self.upscale_blocks = nn.ModuleList(
            [
                UpBlock(256, 128),
                UpBlock(128, 64),
                UpBlock(64, 32),
                UpBlock(32, 16),
            ]
        )

        self.input_conv = nn.Conv2d(3, 16, kernel_size=3, padding="same")
        self.output_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.bridge = BottleNeckBlock(256)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.input_conv(x)

        skip_features = []
        for block in self.downscale_blocks:
            c, x = block(x)
            skip_features.append(c)

        x = self.bridge(x)

        skip_features.reverse()
        for block, skip in zip(self.upscale_blocks, skip_features):
            x = block(x, skip)

        x = self.output_conv(x)
        x = self.activation(x)
        return x


class DownBlock(nn.Module):
    """
    Module that performs downscaling with residual connections.
    """

    def __init__(self, in_channels, out_channels, padding="same", stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        d = self.conv1(x)
        x = self.bn1(d)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x + d
        p = self.maxpool(x)
        return x, p


class UpBlock(nn.Module):
    """
    Module that performs upscaling after concatenation with skip connections.
    """

    def __init__(self, in_channels, out_channels, padding="same", stride=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(
            in_channels * 2,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class BottleNeckBlock(nn.Module):
    """
    BottleNeckBlock that serves as the UNet bridge.
    """

    def __init__(self, channels, padding="same", strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, "same")
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, "same")
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
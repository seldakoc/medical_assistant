"""Original mateuszbuda U-Net architecture (MIT licensed)."""

from collections import OrderedDict

import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        f = init_features
        self.encoder1 = self._block(in_channels, f, "enc1")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = self._block(f, f * 2, "enc2")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = self._block(f * 2, f * 4, "enc3")
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = self._block(f * 4, f * 8, "enc4")
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = self._block(f * 8, f * 16, "bottleneck")
        self.upconv4 = nn.ConvTranspose2d(f * 16, f * 8, 2, 2)
        self.decoder4 = self._block(f * 16, f * 8, "dec4")
        self.upconv3 = nn.ConvTranspose2d(f * 8, f * 4, 2, 2)
        self.decoder3 = self._block(f * 8, f * 4, "dec3")
        self.upconv2 = nn.ConvTranspose2d(f * 4, f * 2, 2, 2)
        self.decoder2 = self._block(f * 4, f * 2, "dec2")
        self.upconv1 = nn.ConvTranspose2d(f * 2, f, 2, 2)
        self.decoder1 = self._block(f * 2, f, "dec1")
        self.conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.decoder4(torch.cat((self.upconv4(b), e4), dim=1))
        d3 = self.decoder3(torch.cat((self.upconv3(d4), e3), dim=1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), e2), dim=1))
        d1 = self.decoder1(torch.cat((self.upconv1(d2), e1), dim=1))
        return torch.sigmoid(self.conv(d1))

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        return nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv2d(in_channels, features, 3, padding=1, bias=False)),
            (name + "norm1", nn.BatchNorm2d(features)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv2d(features, features, 3, padding=1, bias=False)),
            (name + "norm2", nn.BatchNorm2d(features)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ]))

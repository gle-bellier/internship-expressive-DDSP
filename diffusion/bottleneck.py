import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=4,
                              padding=1,
                              stride=2)

    def forward(self, x):
        out = self.conv(x)
        return out

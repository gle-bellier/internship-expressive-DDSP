import torch
import pytorch_lightning as pl
from torch import nn
from utils import FeatureWiseAffine, FiLM, PositionalEncoding, ConvBlock


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, first=False):
        super().__init__()
        if first:
            self.b1 = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            self.b2 = nn.Sequential(
                ConvBlock(in_channels=in_channels,
                          out_channels=out_channels,
                          dilation=1),
                ConvBlock(in_channels=out_channels,
                          out_channels=out_channels,
                          dilation=dilation),
                ConvBlock(in_channels=out_channels,
                          out_channels=out_channels,
                          dilation=1),
            )

        else:
            self.b1 = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1), nn.AvgPool1d(kernel_size=2))
            self.b2 = nn.Sequential(
                nn.AvgPool1d(kernel_size=2),
                ConvBlock(in_channels=in_channels,
                          out_channels=out_channels,
                          dilation=1),
                ConvBlock(in_channels=out_channels,
                          out_channels=out_channels,
                          dilation=dilation),
                ConvBlock(in_channels=out_channels,
                          out_channels=out_channels,
                          dilation=1),
            )

        self.lr = nn.LeakyReLU()

    def forward(self, x):
        out1 = self.b1(x)
        out2 = self.b2(x)

        return out1 + out2

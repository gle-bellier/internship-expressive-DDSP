import torch
import pytorch_lightning as pl
from torch import nn
from utils import FeatureWiseAffine, FiLM, PositionalEncoding, ConvBlock


class UBlock_B(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=out_channels)
        self.conv2 = ConvBlock(in_channels=out_channels,
                               out_channels=out_channels)

    def forward(self, x, film_out):
        x = FeatureWiseAffine(x, film_out)
        x = self.conv1(x)
        x = FeatureWiseAffine(x, film_out)
        out = self.conv2(x)
        return out


class UBlock_A(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LeakyReLU(), nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1))
        self.conv = ConvBlock(in_channels=out_channels,
                              out_channels=out_channels)

    def forward(self, x, film_out):
        x = self.pre(x)
        x = FeatureWiseAffine(x, film_out)
        out = self.conv(x)
        return out


class UBlock_Mid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0))

    def forward(self, x):
        out = self.block(x)
        return out


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.A1 = UBlock_A(in_channels=in_channels, out_channels=out_channels)
        self.A2 = UBlock_Mid(in_channels=in_channels,
                             out_channels=out_channels)
        self.A3 = UBlock_A(in_channels=in_channels, out_channels=out_channels)

        self.B1 = UBlock_B(in_channels=out_channels, out_channels=out_channels)
        self.B2 = UBlock_B(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x, film_out_pitch, film_out_noisy):
        out = self.A1(x, film_out_pitch) + self.A2(x) + self.A3(
            x, film_out_noisy)
        out = out + self.B1(out, film_out_pitch) + self.B2(out, film_out_noisy)

        return out
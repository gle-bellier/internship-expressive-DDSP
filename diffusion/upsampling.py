import torch
import pytorch_lightning as pl
from torch import nn
from utils import FeatureWiseAffine, FiLM, PositionalEncoding, ConvBlock


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.lr = nn.LeakyReLU()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=in_channels,
                               dilation=dilation)

        self.conv2 = ConvBlock(in_channels=in_channels,
                               out_channels=out_channels,
                               dilation=dilation)

        self.fwa1 = FeatureWiseAffine()
        self.fwa2 = FeatureWiseAffine()

    def forward(self, x, film_out_pitch, film_out_noisy):
        x = self.up(x)
        x = self.fwa1(x, film_out_pitch)
        x = self.conv1(x)
        x = self.fwa2(x, film_out_noisy)
        out = self.conv2(x)
        return out


class UBlock_Mid(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, last):
        super().__init__()
        self.last = last
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=dilation,
                              dilation=dilation)

        self.lr = nn.LeakyReLU()

    def forward(self, x):
        if not self.last:
            x = self.up(x)
        out = self.conv(x)
        out = self.lr(out)
        return out


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, last=False):
        super().__init__()
        self.last = last
        self.main = UBlock_Mid(in_channels=in_channels,
                               out_channels=out_channels,
                               dilation=dilation,
                               last=last)

        self.residual = Residual(in_channels=in_channels,
                                 out_channels=out_channels,
                                 dilation=dilation)

    def forward(self, x, film_out_pitch, film_out_noisy):
        out = self.main(x)
        if not self.last:
            out += self.residual(x, film_out_pitch, film_out_noisy)
        return out

import torch
import pytorch_lightning as pl
from torch import nn
from utils import FeatureWiseAffine, FiLM, PositionalEncoding, ConvBlock


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lr = nn.LeakyReLU()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=in_channels)

        self.conv2 = ConvBlock(in_channels=in_channels,
                               out_channels=out_channels)

    def forward(self, x, film_out_pitch, film_out_noisy):
        x = self.up(x)
        x = FeatureWiseAffine()(x, film_out_pitch)
        x = self.conv1
        x = FeatureWiseAffine()(x, film_out_noisy)
        out = self.conv2(x)
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

        self.main = UBlock_Mid(in_channels=in_channels,
                               out_channels=out_channels)
        self.residual = Residual(in_channels=in_channels,
                                 out_channels=out_channels)

    def forward(self, x, film_out_pitch, film_out_noisy):
        print("Input ", x.shape)
        out = self.main(x)
        print("Out main  :", out.shape)
        #out_residual = self.residual(x, film_out_pitch, film_out_noisy)
        #print("Out residual :", out_residual)
        return out
import torch
import pytorch_lightning as pl
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.lr = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        out = self.lr(x)
        return out


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lr = nn.LeakyReLU()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.mp = nn.MaxPool1d(kernel_size=2)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        ctx = torch.clone(x)
        out = self.mp(x)
        return out, ctx


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               stride=1,
                               kernel_size=3,
                               padding=1))

        self.lr = nn.LeakyReLU()

    def add_ctx(self, x, ctx):
        # crop context (y)
        d_shape = (ctx.shape[-1] - x.shape[-1]) // 2
        crop = ctx[:, :, d_shape:d_shape + x.shape[2]]
        #concatenate
        out = torch.cat([x, crop], 1)
        return out

    def forward(self, x, ctx):
        x = self.up_conv(x)
        x = self.add_ctx(x, ctx)
        x = self.conv1(x)
        out = self.conv2(x)
        return out
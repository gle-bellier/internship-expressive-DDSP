import torch
import torch.nn as nn


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
        x = self.lr(x)
        out = self.conv(x)
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
        ctx = torch.clone(x)
        x = self.conv2(x)
        out = self.mp(x)
        return out, ctx


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.gru = nn.GRU(input_size=out_channels,
                          hidden_size=out_channels,
                          batch_first=True)

    def forward(self, x):

        x = self.conv1(x)

        out = self.conv2(x)
        return out


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(2 * out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               stride=1,
                               kernel_size=3,
                               padding=1))
        self.conv_ctx = nn.ConvTranspose1d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           stride=1,
                                           kernel_size=3,
                                           padding=1)

        self.lr = nn.LeakyReLU()

    def add_ctx(self, x, ctx):

        out = torch.cat([x, ctx], 1)
        return out

    def forward(self, x, ctx):
        x = self.up_conv(x)
        ctx = self.conv_ctx(ctx)
        x = self.add_ctx(x, ctx)
        x = self.conv1(x)
        out = self.conv2(x)
        return out

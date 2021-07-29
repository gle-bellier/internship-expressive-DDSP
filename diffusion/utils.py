import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 norm=True,
                 dropout=0.01):
        super().__init__()
        self.norm = norm
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              dilation=dilation,
                              padding=dilation,
                              stride=1)

        self.lr = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        x = self.dp(x)
        out = self.lr(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, n_dim, multiplier=30):
        super().__init__()
        self.n_dim = n_dim
        exponents = 1e-4**torch.linspace(0, 1, n_dim // 2)
        self.register_buffer("exponents", exponents)
        self.multiplier = multiplier

    def forward(self, level):
        level = level.reshape(-1, 1)
        exponents = self.exponents.unsqueeze(0)
        encoding = exponents * level * self.multiplier
        encoding = torch.stack([encoding.sin(), encoding.cos()], -1)
        encoding = encoding.reshape(*encoding.shape[:1], -1)
        return encoding.unsqueeze(-1)


class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.lr = nn.LeakyReLU()
        self.pe = PositionalEncoding(out_channels)
        self.shift_conv = nn.Conv1d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.scale_conv = nn.Conv1d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

    def forward(self, x, noise_level):
        out = self.in_conv(x)
        out = self.lr(out)
        if noise_level is not None:
            pe = self.pe(noise_level)
            out = out + pe
        scale = self.scale_conv(out)
        shift = self.shift_conv(out)
        return scale, shift


class FiLM_RNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.gru = nn.GRU(input_size=out_channels,
                          hidden_size=out_channels,
                          num_layers=num_layers,
                          batch_first=True)

        self.lr = nn.LeakyReLU()
        self.pe = PositionalEncoding(out_channels)
        self.shift_conv = nn.Conv1d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.scale_conv = nn.Conv1d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

    def forward(self, x, noise_level):

        out = x.permute(0, 2, 1)
        out, _ = self.gru(out)
        out = out.permute(0, 2, 1)
        out = self.in_conv(out)

        out = self.lr(out)
        if noise_level is not None:
            pe = self.pe(noise_level)
            out = out + pe
        scale = self.scale_conv(out)
        shift = self.shift_conv(out)
        return scale, shift


class FeatureWiseAffine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, film_out):
        scale, shift = film_out

        out = scale * x + shift
        return out
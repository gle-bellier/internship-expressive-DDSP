import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin


class Identity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, y=None):
        return X.numpy()


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
                                 in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.lr = nn.LeakyReLU()
        self.pe = PositionalEncoding(in_channels)
        self.shift_conv = nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.scale_conv = nn.Conv1d(in_channels,
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


class FeatureWiseAffine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, film_out):
        scale, shift = film_out
        # print("x {}, scale {}, shift {}".format(x.shape, scale.shape,
        #                                         shift.shape))
        out = scale * x + shift
        return out
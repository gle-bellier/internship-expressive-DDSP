import torch
import torch.nn as nn
import pytorch_lightning as pl


class FWAM(nn.Module):
    def forward(self, x, scale, bias):
        return scale * x + bias


class FLConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.ModuleList([
            FWAM(),
            nn.LeakyReLU(.2),
            nn.Conv1d(in_dim, out_dim, 3, padding=1),
        ])

    def forward(self, x, scale, bias):
        for layer in self.net:
            if isinstance(layer, FWAM):
                x = layer(x, scale, bias)
            else:
                x = layer(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_dim, film_dim, out_dim, upsample):
        super().__init__()
        self.upsample = upsample
        self.first_conv = nn.Conv1d(in_dim, out_dim, 3, padding=1)
        self.envelop_flconvs = nn.ModuleList([
            FLConv(out_dim, out_dim),
            FLConv(out_dim, out_dim),
        ])
        self.noise_flconvs = nn.ModuleList([
            FLConv(out_dim, out_dim),
            FLConv(out_dim, out_dim),
        ])
        self.residual_convs = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim, 1),
            nn.Conv1d(out_dim, out_dim, 1),
        ])

    def forward(self, x, envelop_modulation, noise_modulation):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")

        x_residual = self.residual_convs[0](x)

        x = self.first_conv(x)
        for layer in self.envelop_flconvs:
            x = layer(x, *envelop_modulation)

        x = x + x_residual

        x_residual = self.residual_convs[1](x)

        for layer in self.noise_flconvs:
            x = layer(x, *noise_modulation)

        return x + x_residual


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


class FilmBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pe = PositionalEncoding(out_dim)
        self.pre_projection = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 3, padding=1),
            nn.LeakyReLU(.2),
        )
        self.projection = nn.Conv1d(
            out_dim,
            2 * out_dim,
            3,
            padding=1,
            groups=2,
        )

    def forward(self, x, noise_level):
        x = self.pre_projection(x)
        if noise_level is not None:
            x = x + self.pe(noise_level)
        x = self.projection(x)
        return torch.split(x, x.shape[1] // 2, 1)


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        channels = [in_dim] + 3 * [out_dim]
        convs = []
        for i in range(3):
            convs.append(
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=1,
                ))
            convs.append(nn.LeakyReLU(.2))
        self.convs = nn.Sequential(*convs)
        self.res_conv = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        x = x[..., ::2]
        return self.res_conv(x) + self.convs(x)

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from torch import nn
from utils import FiLM, FiLM_RNN, get_padding
from downsampling import DBlock
from upsampling import UBlock
from bottleneck import Bottleneck


class UNet_Diffusion(pl.LightningModule):
    def __init__(self, down_channels, up_channels, down_dilations,
                 up_dilations, scalers):
        super().__init__()
        #self.save_hyperparameters()
        self.down_channels_in = down_channels[:-1]
        self.down_channels_out = down_channels[1:]
        self.down_dilations = down_dilations

        self.up_channels_in = up_channels[:-2]
        self.up_channels_out = up_channels[1:-1]
        self.up_dilations = up_dilations[:-1]

        self.top_channels_in = up_channels[-2]
        self.top_channels_out = up_channels[-1]
        self.top_dilation = up_dilations[-1]

        self.scalers = scalers
        self.val_idx = 0

        first = [True] + [False] * (len(self.down_channels_in) - 1)

        self.down_blocks_pitch = nn.ModuleList([
            DBlock(in_channels=channels_in,
                   out_channels=channels_out,
                   dilation=dilation,
                   first=f) for channels_in, channels_out, dilation, f in zip(
                       self.down_channels_in, self.down_channels_out,
                       self.down_dilations, first)
        ])

        self.down_blocks_noisy = nn.ModuleList([
            DBlock(in_channels=channels_in,
                   out_channels=channels_out,
                   dilation=dilation,
                   first=f) for channels_in, channels_out, dilation, f in zip(
                       self.down_channels_in, self.down_channels_out,
                       self.down_dilations, first)
        ])

        self.films_pitch = nn.ModuleList([
            FiLM(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.down_channels_out,
                                                 self.up_channels_in[::-1])
        ])

        self.films_noisy = nn.ModuleList([
            FiLM(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.down_channels_out,
                                                 self.up_channels_in[::-1])
        ])

        self.up_blocks = nn.ModuleList([
            UBlock(in_channels=channels_in,
                   out_channels=channels_out,
                   dilation=dilation)
            for channels_in, channels_out, dilation in zip(
                self.up_channels_in, self.up_channels_out, self.up_dilations)
        ])

        self.top = UBlock(in_channels=self.top_channels_in,
                          out_channels=self.top_channels_out,
                          dilation=self.top_dilation,
                          last=True)

        self.bottleneck_noisy = Bottleneck(
            in_channels=self.down_channels_out[-1],
            out_channels=self.up_channels_in[0] // 2)

        self.bottleneck_pitch = Bottleneck(
            in_channels=self.down_channels_out[-1],
            out_channels=self.up_channels_in[0] // 2)

    def down_sampling(self, list_blocks, x):
        l_out = []
        for block in list_blocks:
            x = block(x)
            l_out.append(x)
        return l_out

    def up_sampling(self, x, l_film_pitch, l_film_noisy):
        l_film_pitch = l_film_pitch[::-1]
        l_film_noisy = l_film_noisy[::-1]

        for b, p, n in zip(self.up_blocks, l_film_pitch, l_film_noisy):
            x = b(x, p, n)

        out = self.top(x, None, None)
        return out

    def film(self, list_films, l_out, noise_level):
        l_film = []

        for film, out in zip(list_films, l_out):
            l_film.append(film(out, noise_level))

        return l_film

    def cat_hiddens(self, h_pitch, h_noisy):
        hiddens = torch.cat((h_pitch, h_noisy), dim=1)
        out = self.cat_conv(hiddens)
        return out

    def forward(self, noisy, pitch, noise_level):

        l_out_pitch = self.down_sampling(self.down_blocks_pitch, pitch)
        l_out_noisy = self.down_sampling(self.down_blocks_noisy, noisy)

        l_film_pitch = self.film(self.films_pitch, l_out_pitch, noise_level)
        l_film_noisy = self.film(self.films_noisy, l_out_noisy, noise_level)

        out_bottleneck_noisy = self.bottleneck_noisy(l_out_noisy[-1])
        out_bottleneck_pitch = self.bottleneck_pitch(l_out_pitch[-1])

        hiddens = torch.cat((out_bottleneck_pitch, out_bottleneck_noisy),
                            dim=1)

        out = self.up_sampling(hiddens, l_film_pitch, l_film_noisy)

        return out

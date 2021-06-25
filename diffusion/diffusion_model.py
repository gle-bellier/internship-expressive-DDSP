import torch
import pytorch_lightning as pl
from torch import nn
from utils import FeatureWiseAffine, FiLM, PositionalEncoding, ConvBlock
from downsampling import DBlock
from upsampling import UBlock


class DiffusionModel(pl.LightningModule):
    def __init__(self, down_channels, up_channels):
        super().__init__()
        self.save_hyperparameters()
        self.down_channels_in = down_channels[:-1]
        self.down_channels_out = down_channels[1:]

        self.up_channels_in = up_channels[:-1]
        self.up_channels_out = up_channels[1:]

        self.down_blocks_pitch = nn.ModuleList([
            DBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.down_channels_in,
                                                 self.down_channels_out)
        ])

        self.down_blocks_noisy = nn.ModuleList([
            DBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.down_channels_in,
                                                 self.down_channels_out)
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
            UBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.up_channels_in,
                                                 self.up_channels_out)
        ])

    def down_sampling(self, list_blocks, x):
        l_out = []
        for i in range(len(list_blocks)):
            x = list_blocks[i](x)
            l_out = l_out + [x]
        return l_out

    def up_sampling(self, x, l_film_pitch, l_film_noisy):
        l_film_pitch = l_film_pitch[::-1]
        l_film_noisy = l_film_noisy[::-1]

        for i in range(len(self.up_blocks)):
            print(" ROund {} -> {}".format(self.up_channels_in[i],
                                           self.up_channels_out[i]))
            x = self.up_blocks[i](x, l_film_pitch[i], l_film_noisy[i])
        return x

    def film(self, list_films, l_out, noise_level):
        l_film = []
        for i in range(len(list_films)):
            f = list_films[i](l_out[i], noise_level)
            l_film = l_film + [f]
        return l_film

    def cat_hiddens(self, h_pitch, h_noisy):
        hiddens = torch.cat((h_pitch, h_noisy), dim=1)
        out = nn.Conv1d(in_channels=self.down_channels_out[-1] * 2,
                        out_channels=self.up_channels_in[0],
                        kernel_size=3,
                        stride=1,
                        padding=1)(hiddens)
        return out

    def forward(self, pitch, noisy, noise_level):

        l_out_pitch = self.down_sampling(self.down_blocks_pitch, pitch)
        l_out_noisy = self.down_sampling(self.down_blocks_noisy, noisy)

        print("Pitch out")
        for elt in l_out_pitch:
            print(elt.shape)

        l_film_pitch = self.film(self.films_pitch, l_out_pitch, None)
        l_film_noisy = self.film(self.films_noisy, l_out_noisy, noise_level)

        print("FILM out")
        for elt in l_film_pitch:
            print(elt[0].shape)

        hiddens = self.cat_hiddens(l_out_pitch[-1], l_out_noisy[-1])
        out = self.up_sampling(hiddens, l_film_pitch, l_film_noisy)

        return out


if __name__ == "__main__":

    noisy = torch.randn(1, 2, 64)  # B x C x T
    pitch = torch.randn(1, 2, 64)

    noise_level = torch.tensor(0.3)

    down_channels = [2, 4, 8, 16]
    up_channels = [16, 8, 4, 2]

    model = DiffusionModel(down_channels, up_channels)
    model(pitch, noisy, noise_level)

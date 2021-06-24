import torch
import pytorch_lightning as pl
from torch import nn
from utils import FeatureWiseAffine, FiLM, PositionalEncoding, ConvBlock
from downsampling import DBlock
from upsampling import UBlock


class DiffusionModel(pl.LightningModule):
    def __init__(self, channels):
        super().__init__()
        self.save_hyperparameters()
        self.channels = channels

    def down_sampling(self, x):
        l_out = []
        for i in range(len(self.channels) - 1):
            print("DownBlock {} : {} -> {}".format(i, self.channels[i],
                                                   self.channels[i + 1]))
            x = DBlock(in_channels=self.channels[i],
                       out_channels=self.channels[i + 1])(x)
            l_out = [x] + l_out
        return l_out

    def up_sampling(self, l_film_pitch, l_film_noisy):
        reverse = self.channels[::-1]  # reverse without the bottleneck channel
        for i in range(len(reverse) - 1):
            print("UpBlock {} : {} -> {}".format(i, reverse[i],
                                                 reverse[i + 1]))
            x = UBlock(in_channels=reverse[i],
                       out_channels=reverse[i + 1])(x, l_film_pitch[i],
                                                    l_film_noisy[i])
        return x

    def forward(self, pitch, noisy):

        l_out_pitch = self.down_sampling(pitch)
        l_out_noisy = self.down_sampling(noisy)

        print(len(l_out_pitch))
        # l_film_pitch = 0.0
        # l_film_noisy = 0.0

        # out = self.up_sampling(l_film_pitch, l_film_noisy)
        # return out


if __name__ == "__main__":

    noisy = torch.randn(1, 2, 20)  # B x C x T
    pitch = torch.randn(1, 2, 20)

    channels = [2, 4, 8, 126]
    model = DiffusionModel(channels)
    model(pitch, noisy)

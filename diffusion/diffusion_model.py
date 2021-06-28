import torch
import pytorch_lightning as pl
from torch import nn
from utils import FiLM, Identity
from downsampling import DBlock
from upsampling import UBlock
from diffusion import DiffusionModel
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from diffusion_dataset import DiffusionDataset


class UNet_Diffusion(pl.LightningModule, DiffusionModel):
    def __init__(self, down_channels, up_channels, scalers):
        super().__init__()
        self.save_hyperparameters()
        self.down_channels_in = down_channels[:-1]
        self.down_channels_out = down_channels[1:]

        self.up_channels_in = up_channels[:-1]
        self.up_channels_out = up_channels[1:]

        self.scalers = scalers

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

        self.cat_conv = nn.Conv1d(in_channels=self.down_channels_out[-1] * 2,
                                  out_channels=self.up_channels_in[0],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

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
        out = self.cat_conv(hiddens)
        return out

    def neural_pass(self, x, cdt, noise_level):

        # permute from B, T, C -> B, C, T
        x = x.permute(0, 2, 1)

        pitch, noisy = torch.split(x, 1, 1)

        l_out_pitch = self.down_sampling(self.down_blocks_pitch, pitch)
        l_out_noisy = self.down_sampling(self.down_blocks_noisy, noisy)

        l_film_pitch = self.film(self.films_pitch, l_out_pitch, None)
        l_film_noisy = self.film(self.films_noisy, l_out_noisy, noise_level)

        hiddens = self.cat_hiddens(l_out_pitch[-1], l_out_noisy[-1])
        out = self.up_sampling(hiddens, l_film_pitch, l_film_noisy)

        # permute from B, C, T -> B, T, C
        out = out.permute(0, 2, 1)

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, None)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log("val_loss", loss)

    def post_process(self, out):
        out = out / 2 + .5  # change range

        f0, l0 = out.split(1, 1)

        # Inverse transforms


if __name__ == "__main__":

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_total")],
        max_epochs=10000,
    )
    list_transforms = [
        (MinMaxScaler, ),
        (QuantileTransformer, 30),
    ]

    dataset = DiffusionDataset(list_transforms=list_transforms)
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])

    down_channels = [2, 16, 64, 256]
    up_channels = [256, 64, 16, 2]

    model = UNet_Diffusion(scalers=dataset.scalers)

    trainer.fit(
        model,
        DataLoader(train, 32, True),
        DataLoader(val, 32),
    )

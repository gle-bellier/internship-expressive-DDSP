import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from torch import nn
from utils import FiLM, Identity
from downsampling import DBlock
from upsampling import UBlock
from diffusion import DiffusionModel
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from diffusion_dataset import DiffusionDataset
import matplotlib.pyplot as plt
import math


class UNet_Diffusion(pl.LightningModule, DiffusionModel):
    def __init__(self, down_channels, up_channels, down_dilations,
                 up_dilations, scalers, ddsp):
        super().__init__()
        #self.save_hyperparameters()
        self.down_channels_in = down_channels[:-1]
        self.down_channels_out = down_channels[1:]
        self.down_dilations = down_dilations

        self.up_channels_in = up_channels[:-1]
        self.up_channels_out = up_channels[1:]
        self.up_dilations = up_dilations

        self.scalers = scalers
        self.ddsp = ddsp
        self.val_idx = 0

        self.down_blocks_pitch = nn.ModuleList([
            DBlock(in_channels=channels_in,
                   out_channels=channels_out,
                   dilation=dilation)
            for channels_in, channels_out, dilation in zip(
                self.down_channels_in, self.down_channels_out,
                self.down_dilations)
        ])

        self.down_blocks_noisy = nn.ModuleList([
            DBlock(in_channels=channels_in,
                   out_channels=channels_out,
                   dilation=dilation)
            for channels_in, channels_out, dilation in zip(
                self.down_channels_in, self.down_channels_out,
                self.down_dilations)
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
        noisy = x.permute(0, 2, 1)
        pitch = cdt.permute(0, 2, 1)

        l_out_pitch = self.down_sampling(self.down_blocks_pitch, pitch)
        l_out_noisy = self.down_sampling(self.down_blocks_noisy, noisy)

        l_film_pitch = self.film(self.films_pitch, l_out_pitch, noise_level)
        l_film_noisy = self.film(self.films_noisy, l_out_noisy, noise_level)

        hiddens = self.cat_hiddens(l_out_pitch[-1], l_out_noisy[-1])
        out = self.up_sampling(hiddens, l_film_pitch, l_film_noisy)

        # permute from B, C, T -> B, T, C
        out = out.permute(0, 2, 1)

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    def training_step(self, batch, batch_idx):
        model_input, cdt = batch
        loss = self.compute_loss(model_input, cdt)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        # loss = self.compute_loss(batch, batch_idx) Why ??

        model_input, cdt = batch
        loss = self.compute_loss(model_input, cdt)
        self.log("val_loss", loss)

        # returns cdt for validation end epoch
        return cdt

    def post_process(self, out):

        # change range [-1, 1] -> [0, 1]
        out = out / 2 + .5

        f0, l0 = torch.split(out, 1, -1)
        f0 = f0.reshape(-1, 1).cpu().numpy()
        l0 = l0.reshape(-1, 1).cpu().numpy()

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0).reshape(-1)
        l0 = self.scalers[1].inverse_transform(l0).reshape(-1)

        return f0, l0

    def validation_epoch_end(self, cdt):

        # test for last cdt
        cdt = cdt[-1]

        self.val_idx += 1

        if self.val_idx % 50:
            return

        device = next(iter(self.parameters())).device

        out = self.partial_denoising(cdt, cdt, 30)

        f0, lo = out[0].split(1, -1)

        plt.plot(f0.cpu())
        self.logger.experiment.add_figure("pitch RAW", plt.gcf(), self.val_idx)
        plt.plot(lo.cpu())
        self.logger.experiment.add_figure("loudness RAW", plt.gcf(),
                                          self.val_idx)

        # select first elt :

        f0, lo = self.post_process(out[0])

        plt.plot(f0)
        self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)
        plt.plot(lo)
        self.logger.experiment.add_figure("loudness", plt.gcf(), self.val_idx)

        if self.ddsp is not None:
            f0 = torch.from_numpy(f0).float().reshape(1, -1, 1).to("cuda")
            lo = torch.from_numpy(lo).float().reshape(1, -1, 1).to("cuda")
            signal = self.ddsp(f0, lo)
            signal = signal.reshape(-1).cpu().numpy()

            self.logger.experiment.add_audio(
                "generation",
                signal,
                self.val_idx,
                16000,
            )

    @torch.no_grad()
    def sample(self, x, cdt):
        x = torch.randn_like(x)
        for i in range(self.n_step)[::-1]:
            x = self.inverse_dynamics(x, cdt, i)
        return x

    @torch.no_grad()
    def partial_denoising(self, x, cdt, n_step):
        noise_level = self.sqrt_alph_cum_prev[n_step]
        eps = torch.randn_like(x)
        x = noise_level * x
        x = x + math.sqrt(1 - noise_level**2) * eps

        for i in range(n_step)[::-1]:
            x = self.inverse_dynamics(x, cdt, i)
        return x


if __name__ == "__main__":
    tb_logger = pl_loggers.TensorBoardLogger('logs/diffusion/')

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
        max_epochs=100000,
        logger=tb_logger)
    list_transforms = [
        (MinMaxScaler, ),
        (QuantileTransformer, 30),
    ]

    dataset = DiffusionDataset(list_transforms=list_transforms)
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])

    down_channels = [2, 16, 256, 512, 1024]
    up_channels = [1024, 512, 256, 16, 2]
    down_dilations = [2, 4, 6, 8]
    up_dilations = [2, 3, 6, 9]

    ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

    model = UNet_Diffusion(scalers=dataset.scalers,
                           down_channels=down_channels,
                           up_channels=up_channels,
                           down_dilations=down_dilations,
                           up_dilations=up_dilations,
                           ddsp=ddsp)

    model.set_noise_schedule()

    trainer.fit(
        model,
        DataLoader(train, 64, True),
        DataLoader(val, 64),
    )

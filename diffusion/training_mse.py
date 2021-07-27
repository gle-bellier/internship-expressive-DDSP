import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from torch import nn
from utils import FiLM
from downsampling import DBlock
from upsampling import UBlock
from diffusion_mse import DiffusionModel
from torch.utils.data import DataLoader, Dataset, random_split
from model import UNet_Diffusion

from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from transforms import PitchTransformer, LoudnessTransformer
from diffusion_dataset import DiffusionDataset
import matplotlib.pyplot as plt
import math

import warnings

warnings.filterwarnings('ignore')


class Network(pl.LightningModule, DiffusionModel):
    def __init__(self, down_channels, up_channels, down_dilations,
                 up_dilations, scalers):
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet_Diffusion(scalers=scalers,
                                    down_channels=down_channels,
                                    up_channels=up_channels,
                                    down_dilations=down_dilations,
                                    up_dilations=up_dilations)

        self.scalers = scalers
        self.ddsp = None
        self.val_idx = 0

    def neural_pass(self, x, cdt, noise_level):

        # permute from B, T, C -> B, C, T
        noisy = x.permute(0, 2, 1)
        pitch = cdt.permute(0, 2, 1)

        out = self.model(noisy, pitch, noise_level)
        out = out.permute(0, 2, 1)

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=1e-4)  #weight_decay=0.1)

    def training_step(self, batch, batch_idx):
        model_input, cdt = batch
        diffusion_loss, mse_loss = self.compute_loss(model_input, cdt)
        loss = diffusion_loss + mse_loss

        self.log("diffusion_loss", diffusion_loss)
        self.log("mse_loss", mse_loss)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        # loss = self.compute_loss(batch, batch_idx) Why ??

        model_input, cdt = batch
        diffusion_loss, mse_loss = self.compute_loss(model_input, cdt)
        loss = diffusion_loss + mse_loss

        self.log("val_diffusion_loss", diffusion_loss)
        self.log("val_mse_loss", mse_loss)
        self.log("val_loss", loss)

        # returns cdt for validation end epoch
        return cdt

    def post_process(self, out):

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

        if self.val_idx % 100:
            return

        device = next(iter(self.parameters())).device

        out = self.sample(cdt, cdt)
        f0, lo = out[0].split(1, -1)

        midi_f0, midi_lo = cdt[0].split(1, -1)

        plt.plot(midi_f0.cpu())
        plt.plot(f0.cpu())
        self.logger.experiment.add_figure("pitch RAW", plt.gcf(), self.val_idx)
        plt.plot(midi_lo.cpu())
        plt.plot(lo.cpu())
        self.logger.experiment.add_figure("loudness RAW", plt.gcf(),
                                          self.val_idx)

        # select first elt :

        f0, lo = self.post_process(out[0])
        midi_f0, midi_lo = self.post_process(cdt[0])

        plt.plot(midi_f0)
        plt.plot(f0)
        self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)
        plt.plot(midi_lo)
        plt.plot(lo)
        self.logger.experiment.add_figure("loudness", plt.gcf(), self.val_idx)

        if self.ddsp is not None:
            f0 = torch.from_numpy(f0).float().reshape(1, -1, 1).to(device)
            lo = torch.from_numpy(lo).float().reshape(1, -1, 1).to(device)
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
    tb_logger = pl_loggers.TensorBoardLogger('logs/diffusion/quantile/')

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
        max_epochs=100000,
        logger=tb_logger)

    list_transforms = [
        (PitchTransformer, {}),
        (LoudnessTransformer, {}),
    ]
    dataset = DiffusionDataset(list_transforms=list_transforms)
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])

    down_channels = [2, 8, 64, 128, 256]
    up_channels = [256, 128, 64, 16, 8,
                   2]  # one more : last up_block without film
    down_dilations = [1, 1, 2, 4, 4]
    up_dilations = [1, 1, 3, 3, 9, 9]

    model = Network(scalers=dataset.scalers,
                    down_channels=down_channels,
                    up_channels=up_channels,
                    down_dilations=down_dilations,
                    up_dilations=up_dilations)

    model.ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()
    model.set_noise_schedule(init=torch.linspace,
                             init_kwargs={
                                 "steps": 100,
                                 "start": 1e-6,
                                 "end": 1e-2
                             })

    trainer.fit(
        model,
        DataLoader(train, 64, True),
        DataLoader(val, 64),
    )
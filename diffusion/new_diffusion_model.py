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
    def __init__(self, down_channels, up_channels, scalers, ddsp):
        self.conv_down = nn.Conv1d(down_channels[0], down_channels[1], 3, 1, 1)
        self.conv_mid = nn.Conv1d(down_channels[1], up_channels[1], 3, 1, 1)
        self.conv_up = nn.Conv1d(up_channels[1], up_channels[0], 3, 1, 1)

    def neural_pass(self, x, cdt, noise_level):

        # permute from B, T, C -> B, C, T
        x = x.permute(0, 2, 1)

        x = self.conv_down(x)
        x = self.conv_mid(x)
        out = self.conv_up(x)

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

        out = self.sampling(cdt, cdt, 20)

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

    down_channels = [2, 16]
    up_channels = [32, 2]

    ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

    model = UNet_Diffusion(scalers=dataset.scalers,
                           down_channels=down_channels,
                           up_channels=up_channels,
                           ddsp=ddsp)

    model.set_noise_schedule()

    trainer.fit(
        model,
        DataLoader(train, 32, True),
        DataLoader(val, 32),
    )

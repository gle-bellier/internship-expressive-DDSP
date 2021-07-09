import torch
import torch.nn as nn
from diffusion import DiffusionModel
from utils import *
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import math
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os


class UNet(pl.LightningModule, DiffusionModel):
    def __init__(self, channels):
        super().__init__()
        #self.save_hyperparameters()

        down_channels = channels
        up_channels = channels[::-1]

        self.down_channels_in = down_channels[:-1]
        self.down_channels_out = down_channels[1:]

        self.up_channels_in = up_channels[:-1]
        self.up_channels_out = up_channels[1:]

        self.val_idx = 0

        self.down_blocks = nn.ModuleList([
            DBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.down_channels_in,
                                                 self.down_channels_out)
        ])

        self.bottleneck = Bottleneck(in_channels=self.down_channels_out[-1],
                                     out_channels=self.up_channels_in[0])

        self.up_blocks = nn.ModuleList([
            UBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.up_channels_in,
                                                 self.up_channels_out)
        ])

    def down_sampling(self, x):
        l_ctx = []
        for i in range(len(self.down_blocks)):
            x, ctx = self.down_blocks[i](x)
            l_ctx = [ctx] + l_ctx

        return x, l_ctx

    def up_sampling(self, x, l_ctx):

        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x, l_ctx[i])
        return x

    def neural_pass(self, x, cdt, noise_level):

        batch_size, channels, width, height = x.size()
        # (b, 1, 28, 28) -> (b,1,  1*28*28)
        x = x.view(batch_size, 1, width * height)
        out, l_ctx = self.down_sampling(x)
        out = self.bottleneck(out)
        out = self.up_sampling(out, l_ctx)
        out = out.view(batch_size, channels, width, height)

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    def training_step(self, batch, batch_idx):
        model_input, target = batch
        loss = self.compute_loss(model_input, target)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        model_input, target = batch
        loss = self.compute_loss(model_input, target)
        self.log("val_loss", loss)

        return (model_input, target)

    @torch.no_grad()
    def partial_denoising(self, x, cdt, n_step):
        noise_level = self.sqrt_alph_cum_prev[n_step]
        eps = torch.randn_like(x)
        x = noise_level * x
        x = x + math.sqrt(1 - noise_level**2) * eps

        for i in range(n_step)[::-1]:
            x = self.inverse_dynamics(x, cdt, i)
        return x

    def validation_epoch_end(self, inputs):

        model_input, target = inputs[-1]  # first elt of last batch
        model_input = model_input[0:1]
        target = target[0:1]
        self.val_idx += 1

        if self.val_idx % 10:
            return

        device = next(iter(self.parameters())).device
        rec = self.partial_denoising(model_input, None, 50)

        plt.imshow(rec.squeeze().cpu(), cmap='gray_r')
        self.logger.experiment.add_figure("rec", plt.gcf(), self.val_idx)

        diff = torch.abs(rec - model_input)
        plt.imshow(diff.squeeze().cpu(), cmap='gray_r')
        self.logger.experiment.add_figure("diff", plt.gcf(), self.val_idx)


if __name__ == "__main__":

    batch_size = 16

    # transforms
    # prepare transforms standard to MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # data
    mnist_train = MNIST(os.getcwd(),
                        train=True,
                        download=True,
                        transform=transform)
    mnist_test = MNIST(os.getcwd(),
                       train=False,
                       download=True,
                       transform=transform)
    train = DataLoader(mnist_train, batch_size=32)
    test = DataLoader(mnist_test, batch_size=32)

    model = UNet(channels=[
        1,
        64,
        128,
    ])

    model.set_noise_schedule()

    check_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    tb_logger = pl_loggers.TensorBoardLogger('logs/diffusion/mnist/')

    trainer = pl.Trainer(gpus=1,
                         max_epochs=1000000,
                         callbacks=[check_callback],
                         logger=tb_logger)
    trainer.fit(model, train, test)

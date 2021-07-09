import torch
import torch.nn as nn
from diffusion import DiffusionModel
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import math
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os


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


class FWAM(nn.Module):
    def forward(self, x, scale, bias):
        return scale * x + bias


class UpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.branch_1 = nn.ModuleList([
            nn.LeakyReLU(.2),
            nn.Conv1d(in_size, out_size, 3, padding=1, dilation=1),
            FWAM(),
            nn.LeakyReLU(.2),
            nn.Conv1d(out_size, out_size, 3, padding=2, dilation=2),
        ])

        self.branch_2 = nn.Conv1d(in_size, out_size, 1)

        self.branch_3 = nn.ModuleList([
            FWAM(),
            nn.LeakyReLU(.2),
            nn.Conv1d(out_size, out_size, 3, padding=4, dilation=4),
            FWAM(),
            nn.LeakyReLU(.2),
            nn.Conv1d(out_size, out_size, 3, padding=8, dilation=8),
        ])

    def forward(self, x, scale, bias):
        x2 = self.branch_2(x)

        for layer in self.branch_1:
            if isinstance(layer, FWAM):
                x = layer(x, scale, bias)
            else:
                x = layer(x)

        x = x + x2
        x3 = x.clone()

        for layer in self.branch_3:
            if isinstance(layer, FWAM):
                x = layer(x, scale, bias)

        return x3 + x


class DownBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.branch_1 = nn.Conv1d(in_size, out_size, 1)
        self.branch_2 = nn.Sequential(
            nn.LeakyReLU(.2),
            nn.Conv1d(in_size, out_size, 3, padding=1, dilation=1),
            nn.LeakyReLU(.2),
            nn.Conv1d(out_size, out_size, 3, padding=3, dilation=3),
            nn.LeakyReLU(.2),
            nn.Conv1d(out_size, out_size, 3, padding=9, dilation=9),
        )

    def forward(self, x):
        return self.branch_1(x) + self.branch_2(x)


class Film(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.in_conv = nn.Conv1d(in_size, out_size, 3, padding=1)
        self.pe = PositionalEncoding(out_size)
        self.out_conv = nn.Conv1d(out_size,
                                  2 * out_size,
                                  3,
                                  padding=1,
                                  groups=2)

    def forward(self, x, level):
        x = self.in_conv(x)
        x = nn.functional.leaky_relu(x, .2)
        x = x + self.pe(level)
        x = self.out_conv(x)
        return torch.split(x, x.shape[1] // 2, 1)


class Model(pl.LightningModule, DiffusionModel):
    def __init__(self):
        super().__init__()

        #self.save_hyperparameters()

        self.val_idx = 0

        self.down_chain = nn.ModuleList([
            nn.Conv1d(2, 128, 3, padding=1),
            DownBlock(128, 128),
            DownBlock(128, 256),
            DownBlock(256, 256),
            DownBlock(256, 512),
        ])

        self.up_chain = nn.ModuleList([
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 128),
            UpBlock(128, 128),
        ])

        self.film_chain = nn.ModuleList([
            Film(128, 128),
            Film(128, 128),
            Film(256, 128),
            Film(256, 256),
        ])

        self.out_conv = nn.Conv1d(128, 2, 3, padding=1)

    def neural_pass(self, x, cdt, noise_level):
        hidden = []
        for layer in self.down_chain:
            x = layer(x)
            hidden.append(x)

        for i in range(len(self.film_chain)):
            hidden[i] = self.film_chain[i](hidden[i], noise_level)

        x = hidden.pop(-1)

        for layer in self.up_chain:
            x = layer(x, *hidden.pop(-1))

        x = self.out_conv(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, None)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log("val_loss", loss)

    def post_process(self, x):
        x = x / 2 + .5
        f0, lo = torch.split(x, 1, 1)
        f0 = mtof(f0.reshape(-1).cpu().numpy() * 127)

        lo = lo.reshape(-1, 1).cpu().numpy()
        lo = self.Q.inverse_transform(lo).reshape(-1)
        return f0, lo

    def validation_epoch_end(self, out):
        self.val_idx += 1

        if self.val_idx % 100:
            return

        device = next(iter(self.parameters())).device
        x = torch.zeros(16, 2, 256).to(device)
        x = self.sample(x)
        f0, lo = self.post_process(x)

        plt.plot(f0)
        self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)
        plt.plot(lo)
        self.logger.experiment.add_figure("loudness", plt.gcf(), self.val_idx)

        if self.ddsp is not None:
            f0 = torch.from_numpy(f0).float().reshape(1, -1, 1)
            lo = torch.from_numpy(lo).float().reshape(1, -1, 1)
            signal = self.ddsp(f0, lo)
            signal = signal.reshape(-1).numpy()

            self.logger.experiment.add_audio(
                "generation",
                signal,
                self.val_idx,
                16000,
            )

    @torch.no_grad()
    def sample(self, x):
        x = torch.randn_like(x)
        for i in range(self.n_step)[::-1]:
            x = self.inverse_dynamics(x, None, i)
        return x

    @torch.no_grad()
    def partial_denoising(self, x, n_step):
        noise_level = self.sqrt_alph_cum_prev[n_step]
        print(f"{noise_level*100:.2f}% of kept")
        eps = torch.randn_like(x)
        x = noise_level * x
        x = x + math.sqrt(1 - noise_level**2) * eps

        for i in range(n_step)[::-1]:
            x = self.inverse_dynamics(x, None, i)
        return x


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
    train = DataLoader(mnist_train, batch_size=64)
    test = DataLoader(mnist_test, batch_size=64)

    model = Model()

    model.set_noise_schedule()

    check_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1000000,
        callbacks=[check_callback],
    )
    trainer.fit(model, train, test)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from baseline_dataset import Baseline_Dataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import pickle
import matplotlib.pyplot as plt
from random import randint, sample
from utils import *


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, norm=True, act=True):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.norm = nn.LayerNorm(out_size) if norm else None
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act:
            x = nn.functional.leaky_relu(x)
        return x


class Model(pl.LightningModule):
    def __init__(self, in_size, hidden_size, out_size, scalers, ddsp):
        super().__init__()
        # self.save_hyperparameters()
        self.scalers = scalers
        self.ddsp = ddsp
        self.val_idx = 0
        self.lr = nn.LeakyReLU()

        self.pre_gru = nn.Sequential(
            LinearBlock(in_size, hidden_size),
            LinearBlock(hidden_size, hidden_size),
        )

        self.gru = nn.GRU(hidden_size,
                          hidden_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

        self.post_gru = nn.Sequential(
            LinearBlock(hidden_size * 2, hidden_size),
            LinearBlock(
                hidden_size,
                out_size,
                norm=False,
                act=False,
            ),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=1e-4,
            weight_decay=.01,
        )

    def forward(self, x):
        x = self.pre_gru(x)
        x = self.gru(x)[0]
        x = self.lr(x)
        x = self.post_gru(x)
        return x

    def split_predictions(self, prediction):
        pred_cents = prediction[..., :100]
        pred_lo = prediction[..., 100:]
        return pred_cents, pred_lo

    def cross_entropy(self, pred_cents, pred_lo, target_cents, target_lo):
        pred_cents = pred_cents.permute(0, 2, 1)
        pred_lo = pred_lo.permute(0, 2, 1)

        target_cents = target_cents.squeeze(-1)
        target_lo = target_lo.squeeze(-1)

        loss_cents = nn.functional.cross_entropy(pred_cents, target_cents)
        loss_lo = nn.functional.cross_entropy(pred_lo, target_lo)

        return loss_cents, loss_lo

    def training_step(self, batch, batch_idx):
        model_input, target = batch
        prediction = self.forward(model_input.float())

        pred_cents, pred_lo = self.split_predictions(prediction)
        target_cents, target_lo = torch.split(target, 1, -1)

        loss_cents, loss_lo = self.cross_entropy(
            pred_cents,
            pred_lo,
            target_cents,
            target_lo,
        )

        loss = loss_cents + loss_lo

        self.log("loss_cents", loss_cents)
        self.log("loss_lo", loss_lo)
        self.log("loss", loss)

        return loss

    def sample_one_hot(self, x):
        n_bin = x.shape[-1]
        sample = torch.distributions.Categorical(logits=x).sample()
        sample = nn.functional.one_hot(sample, n_bin)
        return sample

    @torch.no_grad()
    def generation_loop(self, x):
        context = None

        for i in range(x.shape[1] - 1):
            x_in = x[:, i:i + 1]

            x_out = self.pre_gru(x_in)
            x_out, context = self.gru(x_out, context)
            x_out = self.post_gru(x_out)
            pred_cents, pred_lo = self.split_predictions(x_out)

            cents = self.sample_one_hot(pred_cents)
            lo = self.sample_one_hot(pred_lo)

            cat = torch.cat([cents, lo], -1)
            ndim = cat.shape[-1]

            x[:, i + 1:i + 2, -ndim:] = cat

        pred = x[..., -ndim:]
        pred_cents, pred_lo = self.split_predictions(pred)

        pred_lo = pred_lo[:, 1:]
        pred_cents = pred_cents[:, :-1]

        out = torch.cat([pred_cents, pred_lo], -1)

        return out

    def apply_inverse_transform(self, x, idx):
        scaler = self.scalers[idx]
        x = x.cpu()
        out = scaler.inverse_transform(x.reshape(-1, 1))
        out = torch.from_numpy(out).to("cuda")
        out = out.unsqueeze(0)
        return out.float()

    def post_process(self, out, pitch):
        cents = out[..., :100]
        lo = out[..., 100:]

        cents = torch.argmax(cents, -1) / 100
        lo = torch.argmax(lo, -1) / 120
        pitch = torch.argmax(pitch, -1) / 127

        pitch = self.apply_inverse_transform(pitch.squeeze(0), 0)
        lo = self.apply_inverse_transform(lo.squeeze(0), 1)
        cents = self.apply_inverse_transform(cents.squeeze(0), 2)

        # Change range [0, 1] -> [-0.5, 0.5]
        cents -= 0.5

        f0 = pctof(pitch, cents)

        return f0, lo

    def get_audio(self, model_input, target):

        model_input = model_input.unsqueeze(0).float()
        pitch = model_input[:, 1:, :128]

        out = self.generation_loop(model_input)
        f0, lo = self.post_process(out, pitch)

        y = self.ddsp(f0, lo)

        plt.plot(f0.squeeze().cpu())
        self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)
        plt.plot(lo.squeeze().cpu())
        self.logger.experiment.add_figure("loudness", plt.gcf(), self.val_idx)
        self.logger.experiment.add_audio(
            "generation",
            y,
            self.val_idx,
            16000,
        )

        return y

    def validation_step(self, batch, batch_idx):
        self.val_idx += 1
        model_input, target = batch
        prediction = self.forward(model_input.float())

        pred_cents, pred_lo = self.split_predictions(prediction)
        target_cents, target_lo = torch.split(target, 1, -1)

        loss_cents, loss_lo = self.cross_entropy(
            pred_cents,
            pred_lo,
            target_cents,
            target_lo,
        )

        self.log("val_loss_cents", loss_cents)
        self.log("val_loss_lo", loss_lo)
        self.log("val_loss", loss_cents + loss_lo)

        ## Every 100 epochs : produce audio

        if self.current_epoch % 100 == 0:
            self.get_audio(model_input[0], target[0])


if __name__ == "__main__":

    list_transforms = [
        (MinMaxScaler, ),  # pitch
        (QuantileTransformer, 120),  # lo
        (QuantileTransformer, 100),  # cents
    ]

    dataset = Baseline_Dataset(list_transforms=list_transforms, n_sample=2048)
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])

    down_channels = [2, 16, 512, 1024]
    ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

    model = Model(in_size=472,
                  hidden_size=512,
                  out_size=221,
                  scalers=dataset.scalers,
                  ddsp=ddsp)

    tb_logger = pl_loggers.TensorBoardLogger('logs/baseline/blstm/')
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
        max_epochs=10000,
        logger=tb_logger)

    trainer.fit(
        model,
        DataLoader(train, 32, True),
        DataLoader(val, 32),
    )

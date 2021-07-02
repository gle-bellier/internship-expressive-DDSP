import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from baseline_dataset import Baseline_Dataset
import pytorch_lightning as pl
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
        self.save_hyperparameters()
        self.scalers = scalers
        self.ddsp = ddsp
        self.val_idx = 0

        self.pre_lstm = nn.Sequential(
            LinearBlock(in_size, hidden_size),
            LinearBlock(hidden_size, hidden_size),
        )

        self.lstm = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.post_lstm = nn.Sequential(
            LinearBlock(hidden_size, hidden_size),
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
        x = self.pre_lstm(x)
        x = self.lstm(x)[0]
        x = self.post_lstm(x)
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
        loss_loudness = nn.functional.cross_entropy(pred_lo, target_lo)

        return loss_cents, loss_loudness

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

        self.log("loss_cents", loss_cents)
        self.log("loss_loudness", loss_lo)

        return loss_cents + loss_lo

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

            x_out = self.pre_lstm(x_in)
            x_out, context = self.lstm(x_out, context)
            x_out = self.post_lstm(x_out)
            pred_cents, pred_lo = self.split_predictions(x_out)

            cents = self.sample_one_hot(pred_cents)
            loudness = self.sample_one_hot(pred_lo)

            cat = torch.cat([cents, loudness], -1)
            ndim = cat.shape[-1]

            x[:, i + 1:i + 2, -ndim:] = cat

        pred = x[..., -ndim:]
        pred_cents, pred_lo = self.split_predictions(pred)

        pred_lo = pred_lo[:, 1:]
        pred_cents = pred_cents[:, :-1]

        out = map(lambda x: torch.argmax(x, -1), [pred_cents, pred_lo])

        return list(out)

    def apply_inverse_transform(self, x, idx):
        scaler = self.scalers[idx]
        x = x.cpu()
        out = scaler.inverse_transform(x.reshape(-1, 1))
        out = torch.from_numpy(out).to("cuda")
        out = out.unsqueeze(0)
        return out.float()

    def get_audio(self, model_input, target):

        model_input = model_input.unsqueeze(0).float()
        f0, cents, loudness = self.generation_loop(model_input)
        cents = cents / 100 - .5

        f0 = pctof(f0, cents)

        loudness = loudness / (121 - 1)
        f0 = self.apply_inverse_transform(f0.squeeze(0), 0)
        loudness = self.apply_inverse_transform(loudness.squeeze(0), 1)
        y = self.ddsp(f0, loudness)

        plt.plot(f0.squeeze().cpu())
        self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)
        plt.plot(loudness.squeeze().cpu())
        self.logger.experiment.add_figure("loudness", plt.gcf(), self.val_idx)

        return y

    def validation_step(self, batch, batch_idx):
        self.val_idx += 1
        model_input, target = batch
        prediction = self.forward(model_input.float())

        pred_cents, pred_lo = self.split_predictions(prediction)
        target_cents, target_lo = torch.split(target, 1, -1)

        loss_f0, loss_cents, loss_lo = self.cross_entropy(
            pred_cents,
            pred_lo,
            target_cents,
            target_lo,
        )

        self.log("val_loss_f0", loss_f0)
        self.log("val_loss_loudness", loss_lo)
        self.log("val_total", loss_cents + loss_lo)

        ## Every 100 epochs : produce audio

        if self.current_epoch % 20 == 0:

            audio = self.get_audio(model_input[0], target[0])
            # output audio in Tensorboard
            tb = self.logger.experiment
            n = "Epoch={}".format(self.current_epoch)
            tb.add_audio(tag=n, snd_tensor=audio, sample_rate=16000)


if __name__ == "__main__":

    list_transforms = [
        (MinMaxScaler, ),  # pitch
        (QuantileTransformer, 120),  # loudness
        (QuantileTransformer, 100),  # cents
    ]

    dataset = Baseline_Dataset(list_transforms=list_transforms, n_sample=2048)
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])

    down_channels = [2, 16, 512, 1024]
    ddsp = torch.jit.load("../ddsp_debug_pretrained.ts").eval()

    model = Model(in_size=472,
                  hidden_size=512,
                  out_size=221,
                  scalers=dataset.scalers,
                  ddsp=ddsp)

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
        max_epochs=10000,
    )

    trainer.fit(
        model,
        DataLoader(train, 32, True),
        DataLoader(val, 32),
    )

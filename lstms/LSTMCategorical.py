import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from expressive_dataset import ExpressiveDataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import pickle
import matplotlib.pyplot as plt
from random import randint, sample
from utils import *
import warnings

warnings.filterwarnings('ignore')


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


class ModelCategorical(pl.LightningModule):
    def __init__(self, in_size, hidden_size, out_size, scalers):
        super().__init__()
        self.save_hyperparameters()
        self.scalers = scalers
        self.loudness_nbins = 100
        self.ddsp = None
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
        pred_f0 = prediction[..., :128]
        pred_cents = prediction[..., 128:228]
        pred_loudness = prediction[..., 228:]
        return pred_f0, pred_cents, pred_loudness

    def cross_entropy(self, pred_f0, pred_cents, pred_loudness, target_f0,
                      target_cents, target_loudness):
        pred_f0 = pred_f0.permute(0, 2, 1)
        pred_cents = pred_cents.permute(0, 2, 1)
        pred_loudness = pred_loudness.permute(0, 2, 1)

        target_f0 = target_f0.squeeze(-1)
        target_cents = target_cents.squeeze(-1)
        target_loudness = target_loudness.squeeze(-1)

        loss_f0 = nn.functional.cross_entropy(pred_f0, target_f0)
        loss_cents = nn.functional.cross_entropy(pred_cents, target_cents)
        loss_loudness = nn.functional.cross_entropy(
            pred_loudness,
            target_loudness,
        )

        return loss_f0, loss_cents, loss_loudness

    def training_step(self, batch, batch_idx):
        model_input, target = batch
        prediction = self.forward(model_input.float())

        pred_f0, pred_cents, pred_loudness = self.split_predictions(prediction)
        target_f0, target_cents, target_loudness = torch.split(target, 1, -1)

        loss_f0, loss_cents, loss_loudness = self.cross_entropy(
            pred_f0,
            pred_cents,
            pred_loudness,
            target_f0,
            target_cents,
            target_loudness,
        )

        self.log("loss_f0", loss_f0)
        self.log("loss_cents", loss_cents)
        self.log("loss_loudness", loss_loudness)

        return loss_f0 + loss_cents + loss_loudness

    def sample_one_hot(self, x):
        n_bin = x.shape[-1]
        sample = torch.distributions.Categorical(logits=x).sample()
        sample = nn.functional.one_hot(sample, n_bin)
        return sample

    @torch.no_grad()
    def generation_loop(self, x, infer_pitch=False):
        context = None

        for i in range(x.shape[1] - 1):
            x_in = x[:, i:i + 1]

            x_out = self.pre_lstm(x_in)
            x_out, context = self.lstm(x_out, context)
            x_out = self.post_lstm(x_out)
            pred_f0, pred_cents, pred_loudness = self.split_predictions(x_out)

            if infer_pitch:
                f0 = self.sample_one_hot(pred_f0)
            else:
                f0 = x[:, i + 1:i + 2, :128].float()

            cents = self.sample_one_hot(pred_cents)
            loudness = self.sample_one_hot(pred_loudness)

            cat = torch.cat([f0, cents, loudness], -1)
            ndim = cat.shape[-1]

            x[:, i + 1:i + 2, -ndim:] = cat

        pred = x[..., -ndim:]
        pred_f0, pred_cents, pred_loudness = self.split_predictions(pred)

        return [pred_f0, pred_cents, pred_loudness]

    def get_audio(self, model_input, target):

        model_input = model_input.unsqueeze(0).float()
        pitch, cents, loudness = self.generation_loop(model_input)

        c = torch.argmax(cents, -1, keepdim=True) / 100
        lo = torch.argmax(loudness, -1, keepdim=True) / 120
        p = torch.argmax(pitch, -1, keepdim=True) / 127

        p = self.scalers[0].inverse_transform(p.squeeze(0).cpu())
        lo = self.scalers[1].inverse_transform(lo.squeeze(0).cpu())
        c = self.scalers[2].inverse_transform(c.squeeze(0).cpu())

        # Change range [0, 1] -> [-0.5, 0.5]
        c -= 0.5

        f0 = pctof(p, c)
        f0 = torch.Tensor(f0).to("cuda")
        lo = torch.Tensor(lo).to("cuda")

        y = self.ddsp(f0.unsqueeze(0), lo.unsqueeze(0))

        plt.plot(f0.squeeze().cpu())
        self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)
        plt.plot(lo.squeeze().cpu())
        self.logger.experiment.add_figure("loudness", plt.gcf(), self.val_idx)

        return y

    def validation_step(self, batch, batch_idx):
        self.val_idx += 1
        model_input, target = batch
        prediction = self.forward(model_input.float())

        pred_f0, pred_cents, pred_loudness = self.split_predictions(prediction)
        target_f0, target_cents, target_loudness = torch.split(target, 1, -1)

        loss_f0, loss_cents, loss_loudness = self.cross_entropy(
            pred_f0,
            pred_cents,
            pred_loudness,
            target_f0,
            target_cents,
            target_loudness,
        )

        self.log("val_loss_f0", loss_f0)
        self.log("val_loss_cents", loss_cents)
        self.log("val_loss_loudness", loss_loudness)
        self.log("val_loss", loss_f0 + loss_cents + loss_loudness)

        ## Every 100 epochs : produce audio

        if self.val_idx % 20 == 0:

            audio = self.get_audio(model_input[0], target[0])
            # output audio in Tensorboard
            tb = self.logger.experiment
            n = "Epoch={}".format(self.current_epoch)
            tb.add_audio(tag=n, snd_tensor=audio, sample_rate=16000)


if __name__ == "__main__":

    inst = "flute"

    tb_logger = pl_loggers.TensorBoardLogger(
        'logs/lstm/categorical/{}/'.format(inst))
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
        max_epochs=10000,
        logger=tb_logger)

    list_transforms = [
        (MinMaxScaler, {}),  # pitch
        (QuantileTransformer, {
            "n_quantiles": 120
        }),  # lo
        (QuantileTransformer, {
            "n_quantiles": 100
        }),  # cents
    ]

    dataset = ExpressiveDataset(list_transforms=list_transforms,
                                path="dataset/{}-train.pickle".format(inst))
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len
    train, val = random_split(dataset, [train_len, val_len])

    model = ModelCategorical(598, 1024, 349, scalers=dataset.scalers)
    model.ddsp = torch.jit.load("ddsp_{}_pretrained.ts".format(inst)).eval()

    trainer.fit(
        model,
        DataLoader(train, 16, True),
        DataLoader(val, 16),
    )

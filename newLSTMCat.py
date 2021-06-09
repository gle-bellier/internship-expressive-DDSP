import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pickle
from random import randint


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


class FullModel(pl.LightningModule):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()

        self.pre_lstm = nn.Sequential(
            LinearBlock(in_size, hidden_size),
            LinearBlock(hidden_size, hidden_size),
        )

        self.lstm = nn.LSTM(hidden_size, batch_first=True)

        self.post_lstm = nn.Sequential(
            LinearBlock(hidden_size, hidden_size),
            LinearBlock(
                hidden_size,
                out_size,
                norm=False,
                act=False,
            ),
        )

    def forward(self, x):
        x = self.pre_lstm(x)
        x = self.lstm(x)[0]
        x = self.post_lstm(x)
        return x

    def training_step(self, batch, batch_idx):
        model_input, target = batch
        prediction = self.forward(model_input)

        pred_f0 = prediction[..., :100]
        pred_cents = prediction[..., 100:200]
        pred_loudness = prediction[..., 200:]

        target_f0, target_cents, target_loudness = torch.split(target, 1, -1)

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

        self.log("loss_f0", loss_f0)
        self.log("loss_cents", loss_cents)
        self.log("loss_loudness", loss_loudness)

        return loss_f0 + loss_cents + loss_loudness


class ExpressiveDataset(Dataset):
    def __init__(self, n_sample=2050, n_loudness=30):
        with open("dataset.pickle", "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.n_sample = n_sample
        self.n_loudness = n_loudness

    def __len__(self):
        return self.N // self.n_sample

    def __getitem__(self, idx):
        N = self.n_sample
        jitter = randint(0, N // 10)
        idx += jitter
        idx = max(idx, 0)
        idx = min(idx, len(self) * self.n_sample - self.n_sample)

        u_f0 = self.dataset["u_f0"][idx:idx + self.n_sample]
        u_loudness = self.dataset["u_loudness"][0][idx:idx + self.n_sample]
        e_f0 = self.dataset["e_f0"][idx:idx + self.n_sample]
        e_cents = self.dataset["e_cents"][idx:idx + self.n_sample]
        e_loudness = self.dataset["e_loudness"][0][idx:idx + self.n_sample]

        u_f0 = torch.from_numpy(u_f0).long()
        u_loudness = torch.from_numpy(u_loudness).float()
        e_f0 = torch.from_numpy(e_f0).long()
        e_cents = torch.from_numpy(e_cents).float()
        e_loudness = torch.from_numpy(e_loudness).float()

        u_f0 = nn.functional.one_hot(u_f0, 100)

        u_loudness = ((self.n_loudness - 1) * u_loudness).long()
        u_loudness = nn.functional.one_hot(u_loudness, self.n_loudness)

        e_f0 = nn.functional.one_hot(e_f0, 100)

        e_cents = (99 * e_cents).long()
        e_cents = nn.functional.one_hot(e_cents, 100)

        e_loudness = ((self.n_loudness - 1) * e_loudness).long()
        e_loudness = nn.functional.one_hot(e_loudness, self.n_loudness)

        model_input = torch.cat([
            u_f0[2:],
            u_loudness[2:],
            e_f0[1:-1],
            e_cents[:-2],
            e_loudness[1:-1],
        ], -1)

        target = torch.cat([
            torch.argmax(e_f0[2:], -1, keepdim=True),
            torch.argmax(e_cents[1:-1], -1, keepdim=True),
            torch.argmax(e_loudness[2:], -1, keepdim=True),
        ], -1)

        return model_input, target


if __name__ == "__main__":
    pass
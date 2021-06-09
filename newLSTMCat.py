import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
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
        return x


class FullModel(pl.LightningModule):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.save_hyperparameters()

        self.pre_lstm = nn.Sequential(
            LinearBlock(in_size, hidden_size),
            LinearBlock(hidden_size, hidden_size),
        )

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
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
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = self.pre_lstm(x)
        x = self.lstm(x)[0]
        x = self.post_lstm(x)
        return x

    def split_predictions(self, prediction):
        pred_f0 = prediction[..., :100]
        pred_cents = prediction[..., 100:200]
        pred_loudness = prediction[..., 200:]
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
    def generation_loop(self, x):
        context = None

        for i in range(x.shape[1] - 1):
            x_in = x[:, i:i + 1]

            x_out = self.pre_lstm(x_in)
            x_out, context = self.lstm(x_out, context)
            x_out = self.post_lstm(x_out)
            pred_f0, pred_cents, pred_loudness = self.split_predictions(x_out)

            f0 = self.sample_one_hot(pred_f0)
            cents = self.sample_one_hot(pred_cents)
            loudness = self.sample_one_hot(pred_loudness)

            cat = torch.cat([f0, cents, loudness], -1)
            ndim = cat.shape[-1]

            x[:, i + 1:i + 2, -ndim:] = cat

        pred = x[..., -ndim:]
        pred_f0, pred_cents, pred_loudness = self.split_predictions(pred)

        pred_f0 = pred_f0[:, 1:]
        pred_loudness = pred_loudness[:, 1:]
        pred_cents = pred_cents[:, :-1]

        out = map(lambda x: torch.argmax(x, -1),
                  [pred_f0, pred_cents, pred_loudness])

        return list(out)

    def validation_step(self, batch, batch_idx):
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
        self.log("val_total", loss_f0 + loss_cents + loss_loudness)


class ExpressiveDataset(Dataset):
    def __init__(self, n_sample=2050, n_loudness=30):
        with open("dataset.pickle", "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.n_sample = n_sample
        self.n_loudness = n_loudness

    def unnormalize_loudness(self, x):
        min_lo, max_lo = self.dataset["e_loudness"][1:]
        x = x * (max_lo - min_lo) + min_lo
        return x

    def __len__(self):
        return self.N // self.n_sample

    def __getitem__(self, idx):
        N = self.n_sample
        idx *= N
        
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

        model_input = torch.cat(
            [
                u_f0[2:],
                u_loudness[2:],
                e_f0[1:-1],  # one step behind
                e_cents[:-2],  # two steps behind
                e_loudness[1:-1],  # one step behind
            ],
            -1)

        target = torch.cat([
            torch.argmax(e_f0[2:], -1, keepdim=True),
            torch.argmax(e_cents[1:-1], -1, keepdim=True),
            torch.argmax(e_loudness[2:], -1, keepdim=True),
        ], -1)

        return model_input, target


if __name__ == "__main__":

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_total")],
    )

    dataset = ExpressiveDataset()
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])

    model = FullModel(360, 256, 230)

    trainer.fit(
        model,
        DataLoader(train, 32, True),
        DataLoader(val, 32),
    )

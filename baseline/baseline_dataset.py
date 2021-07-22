import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
import pytorch_lightning as pl
import pickle
from random import randint
from utils import *


class Baseline_Dataset(Dataset):
    def __init__(self,
                 list_transforms,
                 path="dataset/dataset-article.pickle",
                 n_sample=2048,
                 eval=False):
        with open(path, "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.list_transforms = list_transforms
        self.n_sample = n_sample
        self.scalers = self.fit_transforms()
        self.eval = eval

    def fit_transforms(self):
        scalers = []
        # pitch :

        cat = np.concatenate((self.dataset["u_f0"], self.dataset["e_f0"]))
        contour = cat.reshape(-1, 1)
        transform = self.list_transforms[0]
        sc = transform[0]
        sc = sc(*transform[1:]).fit(contour)
        scalers.append(sc)

        # loudness

        contour = self.dataset["e_loudness"]
        contour = contour.reshape(-1, 1)
        transform = self.list_transforms[1]
        sc = transform[0]
        sc = sc(*transform[1:]).fit(contour)
        scalers.append(sc)

        # cents

        contour = self.dataset["e_cents"]
        contour = contour.reshape(-1, 1)
        transform = self.list_transforms[2]
        sc = transform[0]
        sc = sc(*transform[1:]).fit(contour)
        scalers.append(sc)

        return scalers

    def apply_transform(self, x, scaler):
        out = scaler.transform(x.reshape(-1, 1)).squeeze(-1)
        return out

    def post_processing(self, p, c, lo):

        c = torch.argmax(c, -1, keepdim=True) / 100
        lo = torch.argmax(lo, -1, keepdim=True) / 120
        p = torch.argmax(p, -1, keepdim=True) / 127

        p = self.scalers[0].inverse_transform(p.squeeze(0))
        lo = self.scalers[1].inverse_transform(lo.squeeze(0))
        c = self.scalers[2].inverse_transform(c.squeeze(0))

        # Change range [0, 1] -> [-0.5, 0.5]
        c -= 0.5

        f0 = pctof(p, c)

        return f0, lo

    def get_quantized_loudness(self, e_l0, onsets, offsets):
        events = onsets + offsets
        e = torch.abs(events)
        u_l0 = torch.zeros_like(e_l0)

        # get indexes of events
        indexes = (e == 1).nonzero(as_tuple=True)[0]
        start, end = torch.tensor([0]), torch.tensor([e_l0.shape[0] - 1])
        indexes = torch.cat([start, indexes, end], -1)

        for i in range(len(indexes) - 1):
            u_l0[indexes[i]:indexes[i + 1]] = torch.mean(
                e_l0[indexes[i]:indexes[i + 1]])
        return u_l0

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
        e_f0 = self.dataset["e_f0"][idx:idx + self.n_sample]
        e_cents = self.dataset["e_cents"][idx:idx + self.n_sample]
        e_lo = self.dataset["e_loudness"][idx:idx + self.n_sample]
        onsets = self.dataset["onsets"][idx:idx + self.n_sample]
        offsets = self.dataset["offsets"][idx:idx + self.n_sample]
        # Apply transforms :

        u_f0 = self.apply_transform(u_f0, self.scalers[0])
        e_f0 = self.apply_transform(e_f0, self.scalers[0])
        e_lo = self.apply_transform(e_lo, self.scalers[1])
        e_cents = self.apply_transform(e_cents, self.scalers[2])

        u_f0 = torch.from_numpy(u_f0).float()
        e_f0 = torch.from_numpy(e_f0).float()
        e_lo = torch.from_numpy(e_lo).float()
        e_cents = torch.from_numpy(e_cents).float()
        onsets = torch.from_numpy(onsets).float()
        offsets = torch.from_numpy(offsets).float()

        u_lo = self.get_quantized_loudness(e_lo, onsets, offsets)

        u_f0 = (127 * u_f0).long()
        u_f0 = nn.functional.one_hot(u_f0, 128)

        u_lo = (120 * u_lo).long()
        u_lo = nn.functional.one_hot(u_lo, 121)

        e_cents = (99 * e_cents).long()
        e_cents = nn.functional.one_hot(e_cents, 100)

        e_lo = (120 * e_lo).long()
        e_lo = nn.functional.one_hot(e_lo, 121)

        onsets = onsets.reshape(-1, 1)
        offsets = offsets.reshape(-1, 1)

        model_input = torch.cat(
            [
                u_f0[1:],
                u_lo[1:],
                e_cents[:-1],  # one step behind
                e_lo[:-1],  # one step behind
                onsets[:-1],
                offsets[:-1]
            ],
            -1)

        target = torch.cat([
            torch.argmax(e_cents[1:], -1, keepdim=True),
            torch.argmax(e_lo[1:], -1, keepdim=True),
        ], -1)
        if self.eval:
            return model_input, target, onsets, offsets

        return model_input, target

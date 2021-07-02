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
                 n_sample=2050,
                 n_loudness=30):
        with open(path, "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.list_transforms = list_transforms
        self.n_sample = n_sample
        self.n_loudness = n_loudness
        self.scalers = self.fit_transforms()

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
        e_f0 = self.apply_transform(e_f0, self.scalers[2])
        e_cents = self.apply_transform(e_cents, self.scalers[3])
        e_lo = self.apply_transform(e_lo, self.scalers[4])

        u_f0 = torch.from_numpy(u_f0).float()
        e_f0 = torch.from_numpy(e_f0).float()
        e_l0 = torch.from_numpy(e_lo).float()
        onsets = torch.from_numpy(onsets).float()
        offsets = torch.from_numpy(offsets).float()

        u_lo = self.get_quantized_loudness(e_l0, onsets, offsets)

        u_f0 = nn.functional.one_hot(u_f0, 100)

        u_lo = ((self.n_loudness - 1) * u_lo).long()
        u_lo = nn.functional.one_hot(u_lo, self.n_loudness)

        e_f0 = nn.functional.one_hot(e_f0, 100)

        e_cents = (99 * e_cents).long()
        e_cents = nn.functional.one_hot(e_cents, 100)

        e_lo = ((self.n_loudness - 1) * e_lo).long()
        e_lo = nn.functional.one_hot(e_lo, self.n_loudness)

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
            torch.argmax(e_cents[1], -1, keepdim=True),
            torch.argmax(e_lo[1:], -1, keepdim=True),
        ], -1)

        return model_input, target


class ExpressiveDatasetPitchContinuous(Dataset):
    def __init__(self,
                 list_transforms,
                 path="dataset/dataset-augmented.pickle",
                 n_sample=2050,
                 n_loudness=30):
        with open(path, "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.list_transforms = list_transforms
        self.n_sample = n_sample
        self.n_loudness = n_loudness
        self.scalers = self.fit_transforms()

    def fit_transforms(self):
        data = [
            self.dataset["u_f0"], self.dataset["u_loudness"],
            self.dataset["e_f0"], self.dataset["e_cents"],
            self.dataset["e_loudness"]
        ]

        scalers = []
        for i in range(len(data)):
            contour = data[i].reshape(-1, 1)
            transform = self.list_transforms[i]
            sc = transform[0]
            sc = sc(*transform[1:]).fit(contour)
            scalers.append(sc)
        return scalers

    def apply_transform(self, x, scaler):
        out = scaler.transform(x.reshape(-1, 1)).squeeze(-1)
        return out

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
        u_loudness = self.dataset["u_loudness"][idx:idx + self.n_sample]
        e_f0 = self.dataset["e_f0"][idx:idx + self.n_sample]
        e_cents = self.dataset["e_cents"][idx:idx + self.n_sample]
        e_loudness = self.dataset["e_loudness"][idx:idx + self.n_sample]

        # Apply transforms :

        u_f0 = self.apply_transform(u_f0, self.scalers[0])
        u_loudness = self.apply_transform(u_loudness, self.scalers[1])
        e_f0 = self.apply_transform(e_f0, self.scalers[2])
        e_cents = self.apply_transform(e_cents, self.scalers[3])
        e_loudness = self.apply_transform(e_loudness, self.scalers[4])

        u_f0 = torch.from_numpy(u_f0).long()
        u_loudness = torch.from_numpy(u_loudness).float()
        e_f0 = torch.from_numpy(e_f0).long()
        e_cents = torch.from_numpy(e_cents).float()
        e_loudness = torch.from_numpy(e_loudness).float()

        u_loudness = ((self.n_loudness - 1) * u_loudness).long()
        u_loudness = nn.functional.one_hot(u_loudness, self.n_loudness)

        e_loudness = ((self.n_loudness - 1) * e_loudness).long()
        e_loudness = nn.functional.one_hot(e_loudness, self.n_loudness)

        model_input = torch.cat(
            [
                u_f0[2:].unsqueeze(-1),
                u_loudness[2:],
                e_f0[1:-1].unsqueeze(-1),  # one step behind
                e_cents[:-2].unsqueeze(-1),  # two steps behind
                e_loudness[1:-1],  # one step behind
            ],
            -1)

        target = torch.cat([
            e_f0[2:].unsqueeze(-1),
            e_cents[1:-1].unsqueeze(-1),
            torch.argmax(e_loudness[2:], -1, keepdim=True),
        ], -1)

        return model_input, target

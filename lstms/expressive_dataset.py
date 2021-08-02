import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
import pytorch_lightning as pl
import pickle
from random import randint
from utils import *


class ExpressiveDataset(Dataset):
    def __init__(self,
                 list_transforms,
                 path="dataset/dataset-article.pickle",
                 n_sample=2050,
                 eval=False):
        with open(path, "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.list_transforms = list_transforms
        self.n_sample = n_sample
        self.load()
        self.scalers = self.fit_transforms()
        self.transform()
        self.eval = eval
        print("Dataset loaded. Length : {}min".format(self.N // 6000))

    def fit_transforms(self):
        scalers = []
        # pitch :

        cat = np.concatenate((self.u_f0, self.e_f0))
        contour = cat.reshape(-1, 1)
        transform = self.list_transforms[0]
        sc = transform[0]
        sc = sc(**transform[1]).fit(contour)
        scalers.append(sc)

        # loudness

        contour = self.e_lo
        contour = contour.reshape(-1, 1)
        transform = self.list_transforms[1]
        sc = transform[0]
        sc = sc(**transform[1]).fit(contour)
        scalers.append(sc)

        # cents

        contour = self.e_cents
        contour = contour.reshape(-1, 1)
        transform = self.list_transforms[2]
        sc = transform[0]
        sc = sc(**transform[1]).fit(contour)
        scalers.append(sc)

        return scalers

    def apply_transform(self, x, scaler):
        out = scaler.transform(x.reshape(-1, 1)).squeeze(-1)
        return out

    def load(self):
        self.u_f0 = self.dataset["u_f0"]
        self.e_f0 = self.dataset["e_f0"]
        self.e_lo = self.dataset["e_loudness"]
        self.onsets = self.dataset["onsets"]
        self.offsets = self.dataset["offsets"]

        # split pitch and cents :

        self.e_f0, self.e_cents = ftopc(self.e_f0)
        self.u_f0, _ = ftopc(self.u_f0)

        self.e_f0 = np.clip(self.e_f0, 0, 127)
        self.u_f0 = np.clip(self.u_f0, 0, 127)
        self.e_cents = np.clip(self.e_cents, -50, 50)

        # add shift
        self.e_cents += .5

    def transform(self):

        # apply transforms :

        self.u_f0 = self.apply_transform(self.u_f0, self.scalers[0])
        self.e_f0 = self.apply_transform(self.e_f0, self.scalers[0])
        self.e_lo = self.apply_transform(self.e_lo, self.scalers[1])
        self.e_cents = self.apply_transform(self.e_cents, self.scalers[2])

        self.u_f0 = torch.from_numpy(self.u_f0).float()
        self.e_f0 = torch.from_numpy(self.e_f0).float()
        self.e_lo = torch.from_numpy(self.e_lo).float()
        self.e_cents = torch.from_numpy(self.e_cents).float()
        self.onsets = torch.from_numpy(self.onsets).float()
        self.offsets = torch.from_numpy(self.offsets).float()

        # compute u_lo
        self.u_lo = self.get_quantized_loudness(self.e_lo, self.onsets,
                                                self.offsets)

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

        u_f0 = self.u_f0[idx:idx + self.n_sample]
        u_lo = self.u_lo[idx:idx + self.n_sample]
        e_f0 = self.e_f0[idx:idx + self.n_sample]
        e_cents = self.e_cents[idx:idx + self.n_sample]
        e_lo = self.e_lo[idx:idx + self.n_sample]
        onsets = self.onsets[idx:idx + self.n_sample]
        offsets = self.offsets[idx:idx + self.n_sample]

        u_f0 = (127 * u_f0).long()
        u_f0 = nn.functional.one_hot(u_f0, 128)

        u_lo = (120 * u_lo).long()
        u_lo = nn.functional.one_hot(u_lo, 121)

        e_f0 = (127 * e_f0).long()
        e_f0 = nn.functional.one_hot(e_f0, 128)

        e_cents = (99 * e_cents).long()
        e_cents = nn.functional.one_hot(e_cents, 100)

        e_lo = (120 * e_lo).long()
        e_lo = nn.functional.one_hot(e_lo, 121)

        model_input = torch.cat(
            [
                u_f0[2:],
                u_lo[2:],
                e_f0[1:-1],  # one step behind
                e_cents[:-2],  # two steps behind
                e_lo[1:-1],  # one step behind
            ],
            -1)

        target = torch.cat([
            torch.argmax(e_f0[2:], -1, keepdim=True),
            torch.argmax(e_cents[1:-1], -1, keepdim=True),
            torch.argmax(e_lo[2:], -1, keepdim=True)
        ], -1)

        if self.eval:
            return model_input, target, onsets, offsets

        return model_input, target


class ExpressiveDatasetPitchContinuous(Dataset):
    def __init__(self,
                 list_transforms,
                 path="dataset/dataset-article.pickle",
                 n_sample=2050,
                 eval=False):
        with open(path, "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.list_transforms = list_transforms
        self.n_sample = n_sample
        self.load()
        self.scalers = self.fit_transforms()
        self.transform()
        self.eval = eval
        print("Dataset loaded. Length : {}min".format(self.N // 6000))

    def fit_transforms(self):
        scalers = []
        # pitch :

        cat = np.concatenate((self.u_f0, self.e_f0))
        contour = cat.reshape(-1, 1)
        transform = self.list_transforms[0]
        sc = transform[0]
        sc = sc(**transform[1]).fit(contour)
        scalers.append(sc)

        # loudness

        contour = self.e_lo
        contour = contour.reshape(-1, 1)
        transform = self.list_transforms[1]
        sc = transform[0]
        sc = sc(**transform[1]).fit(contour)
        scalers.append(sc)

        # cents

        contour = self.e_cents
        contour = contour.reshape(-1, 1)
        transform = self.list_transforms[2]
        sc = transform[0]
        sc = sc(**transform[1]).fit(contour)
        scalers.append(sc)

        return scalers

    def apply_transform(self, x, scaler):
        out = scaler.transform(x.reshape(-1, 1)).squeeze(-1)
        return out

    def load(self):
        self.u_f0 = self.dataset["u_f0"]
        self.e_f0 = self.dataset["e_f0"]
        self.e_lo = self.dataset["e_loudness"]
        self.onsets = self.dataset["onsets"]
        self.offsets = self.dataset["offsets"]

        # split pitch and cents :

        self.e_f0, self.e_cents = ftopc(self.e_f0)
        self.u_f0, _ = ftopc(self.u_f0)

        self.e_f0 = np.clip(self.e_f0, 0, 127)
        self.u_f0 = np.clip(self.u_f0, 0, 127)
        self.e_cents = np.clip(self.e_cents, -50, 50)

        # add shift
        self.e_cents += .5

    def transform(self):

        # apply transforms :

        self.u_f0 = self.apply_transform(self.u_f0, self.scalers[0])
        self.e_f0 = self.apply_transform(self.e_f0, self.scalers[0])
        self.e_lo = self.apply_transform(self.e_lo, self.scalers[1])
        self.e_cents = self.apply_transform(self.e_cents, self.scalers[2])

        self.u_f0 = torch.from_numpy(self.u_f0).float()
        self.e_f0 = torch.from_numpy(self.e_f0).float()
        self.e_lo = torch.from_numpy(self.e_lo).float()
        self.e_cents = torch.from_numpy(self.e_cents).float()
        self.onsets = torch.from_numpy(self.onsets).float()
        self.offsets = torch.from_numpy(self.offsets).float()

        # compute u_lo
        self.u_lo = self.get_quantized_loudness(self.e_lo, self.onsets,
                                                self.offsets)

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

        u_f0 = self.u_f0[idx:idx + self.n_sample]
        u_lo = self.u_lo[idx:idx + self.n_sample]
        e_f0 = self.e_f0[idx:idx + self.n_sample]
        e_cents = self.e_cents[idx:idx + self.n_sample]
        e_lo = self.e_lo[idx:idx + self.n_sample]
        onsets = self.onsets[idx:idx + self.n_sample]
        offsets = self.offsets[idx:idx + self.n_sample]

        u_f0 = u_f0.unsqueeze(-1)

        u_lo = (120 * u_lo).long()
        u_lo = nn.functional.one_hot(u_lo, 121)

        e_cents = e_cents.unsqueeze(-1)
        e_f0 = e_f0.unsqueeze(-1)

        e_lo = (120 * e_lo).long()
        e_lo = nn.functional.one_hot(e_lo, 121)

        model_input = torch.cat(
            [
                u_f0[2:],
                u_lo[2:],
                e_f0[1:-1],  # one step behind
                e_cents[:-2],  # two steps behind
                e_lo[1:-1],  # one step behind
            ],
            -1)

        target = torch.cat([
            e_f0[2:], e_cents[1:-1],
            torch.argmax(e_lo[2:], -1, keepdim=True)
        ], -1)

        if self.eval:
            return model_input, target, onsets, offsets

        return model_input, target
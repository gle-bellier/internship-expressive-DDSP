import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import pytorch_lightning as pl
import pickle
import numpy as np
from random import randint


class UNet_Dataset(Dataset):
    def __init__(self,
                 instrument,
                 data_augmentation=False,
                 type_set="train",
                 n_sample=2048,
                 list_transforms=None,
                 eval=False):
        if eval:
            path = "dataset/test-set.pickle"
            print("Eval dataset file used : {}".format(path))
            with open(path, "rb") as dataset:
                dataset = pickle.load(dataset)

        else:
            da = "-da" if data_augmentation else ""
            path = "dataset/{}-{}{}.pickle".format(instrument, type_set, da)
            print("{} dataset file used : {}".format(type_set, path))
            with open(path, "rb") as dataset:
                dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.n_sample = n_sample
        self.list_transforms = list_transforms

        self.scalers = self.fit_transforms()
        self.transform()
        self.eval = eval
        print("Dataset loaded. Length : {}min".format(self.N // 6000))

    def fit_transforms(self):
        scalers = []

        # pitch :
        if self.eval:
            cat = self.dataset["u_f0"]
        else:
            cat = np.concatenate((self.dataset["u_f0"], self.dataset["e_f0"]))
        contour = cat.reshape(-1, 1)

        transform = self.list_transforms[0]
        sc = transform[0]
        sc = sc(**transform[1]).fit(contour)
        scalers.append(sc)

        # loudness
        if self.eval:
            contour = self.dataset["u_f0"]
        else:
            contour = self.dataset["e_loudness"]
        contour = contour.reshape(-1, 1)
        transform = self.list_transforms[1]
        sc = transform[0]
        sc = sc(**transform[1]).fit(contour)
        scalers.append(sc)

        return scalers

    def transform(self):
        self.u_f0 = self.dataset["u_f0"]
        self.u_lo = self.dataset["u_loudness"]

        self.e_f0 = self.dataset["e_f0"]
        self.e_lo = self.dataset["e_loudness"]

        self.onsets = self.dataset["onsets"]
        self.offsets = self.dataset["offsets"]

        self.onsets = torch.from_numpy(self.onsets).float()
        self.offsets = torch.from_numpy(self.offsets).float()

        # Apply transforms :

        self.u_f0 = self.apply_transform(self.u_f0, self.scalers[0])
        self.e_f0 = self.apply_transform(self.e_f0, self.scalers[0])
        self.u_lo = self.apply_transform(self.u_lo, self.scalers[1])
        self.e_lo = self.apply_transform(self.e_lo, self.scalers[1])

        self.u_f0 = torch.from_numpy(self.u_f0).float()
        self.e_f0 = torch.from_numpy(self.e_f0).float()
        self.e_lo = torch.from_numpy(self.e_lo).float()

        if self.eval:
            self.u_lo = torch.from_numpy(self.u_lo).float()
        else:
            self.u_lo = self.get_quantized_loudness(self.e_lo, self.onsets,
                                                    self.offsets)

    def apply_transform(self, x, scaler):
        out = scaler.transform(x.reshape(-1, 1)).squeeze(-1)
        return out

    def inverse_transform(self, x):

        f0, lo = torch.split(x, 1, -1)
        f0 = f0.reshape(-1, 1).cpu().numpy()
        lo = lo.reshape(-1, 1).cpu().numpy()

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0).reshape(-1)
        lo = self.scalers[1].inverse_transform(lo).reshape(-1)

        return f0, lo

    def get_quantized_loudness(self, e_l0, onsets, offsets):
        e = torch.abs(onsets + offsets)
        e = torch.tensor(
            [e[i] if e[i + 1] != 1 else 0 for i in range(len(e) - 1)])
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

        # add jitter during training only
        if not self.eval:
            idx += randint(0, N // 10)
        idx = max(idx, 0)
        idx = min(idx, len(self) * self.n_sample - self.n_sample)

        s_u_f0 = self.u_f0[idx:idx + self.n_sample]
        s_u_lo = self.u_lo[idx:idx + self.n_sample]
        s_e_f0 = self.e_f0[idx:idx + self.n_sample]
        s_e_lo = self.e_lo[idx:idx + self.n_sample]
        s_onsets = self.onsets[idx:idx + self.n_sample]
        s_offsets = self.offsets[idx:idx + self.n_sample]

        model_input = torch.cat([
            s_u_f0.unsqueeze(-1),
            s_u_lo.unsqueeze(-1),
        ], -1)

        target = torch.cat([
            s_e_f0.unsqueeze(-1),
            s_e_lo.unsqueeze(-1),
        ], -1)

        if self.eval:
            return model_input, target, s_onsets, s_offsets

        return model_input, target
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import pytorch_lightning as pl
import pickle
import numpy as np
from random import randint


class DiffusionDataset(Dataset):
    def __init__(self,
                 path="dataset/dataset-diffusion.pickle",
                 n_sample=2048,
                 n_loudness=30,
                 list_transforms=None,
                 eval=False):

        with open(path, "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.n_sample = n_sample
        self.n_loudness = n_loudness
        self.list_transforms = list_transforms

        self.scalers = self.fit_transforms()
        self.eval = eval

    def mtof(self, m):
        return 440 * 2**((m - 69) / 12)

    def ftom(self, f):
        return 12 * np.log2(f / 440) + 69

    def fit_transforms(self):
        scalers = []

        # pitch :

        cat = np.concatenate((self.dataset["u_f0"], self.dataset["e_f0"]))
        contour = cat.reshape(-1, 1)

        # go log scale :
        # contour = self.ftom(contour)

        transform = self.list_transforms[0]

        sc = transform[0]
        sc = sc(**transform[1]).fit(contour)
        scalers.append(sc)

        # loudness

        contour = self.dataset["e_loudness"]
        contour = contour.reshape(-1, 1)
        transform = self.list_transforms[1]
        sc = transform[0]
        sc = sc(*transform[1:]).fit(contour)
        scalers.append(sc)

        return scalers

    def apply_transform(self, x, scaler):
        out = scaler.transform(x.reshape(-1, 1)).squeeze(-1)
        return out

    def inverse_transform(self, x):
        # change range [-1, 1] -> [0, 1]
        x = x / 2 + .5

        f0, lo = torch.split(x, 1, -1)
        f0 = f0.reshape(-1, 1).cpu().numpy()
        lo = lo.reshape(-1, 1).cpu().numpy()

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0).reshape(-1)
        lo = self.scalers[1].inverse_transform(lo).reshape(-1)

        #f0 = self.mtof(f0)
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

        jitter = randint(0, N // 10)
        idx += jitter
        idx = max(idx, 0)
        idx = min(idx, len(self) * self.n_sample - self.n_sample)

        u_f0 = self.dataset["u_f0"][idx:idx + self.n_sample]
        e_f0 = self.dataset["e_f0"][idx:idx + self.n_sample]
        e_l0 = self.dataset["e_loudness"][idx:idx + self.n_sample]
        onsets = self.dataset["onsets"][idx:idx + self.n_sample]
        offsets = self.dataset["offsets"][idx:idx + self.n_sample]

        # Go log scale

        # u_f0 = self.ftom(u_f0)
        # e_f0 = self.ftom(e_f0)

        # Apply transforms :

        u_f0 = self.apply_transform(u_f0, self.scalers[0])
        e_f0 = self.apply_transform(e_f0, self.scalers[0])
        e_l0 = self.apply_transform(e_l0, self.scalers[1])

        u_f0 = torch.from_numpy(u_f0).float()
        e_f0 = torch.from_numpy(e_f0).float()
        e_l0 = torch.from_numpy(e_l0).float()
        onsets = torch.from_numpy(onsets).float()
        offsets = torch.from_numpy(offsets).float()

        u_l0 = self.get_quantized_loudness(e_l0, onsets, offsets)

        # Change ranges from [0, 1] -> [-1, 1]

        u_f0 = 2 * (u_f0 - .5)
        u_l0 = 2 * (u_l0 - .5)
        e_f0 = 2 * (e_f0 - .5)
        e_l0 = 2 * (e_l0 - .5)

        model_input = torch.cat([
            e_f0.unsqueeze(-1),
            e_l0.unsqueeze(-1),
        ], -1)

        cdt = torch.cat([
            u_f0.unsqueeze(-1),
            u_l0.unsqueeze(-1),
        ], -1)
        if self.eval:
            return model_input, cdt, onsets, offsets

        return model_input, cdt
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
                 path="dataset-diffusion.pickle",
                 n_sample=2048,
                 n_loudness=30,
                 list_transforms=None):

        with open(path, "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.n_sample = n_sample
        self.n_loudness = n_loudness
        if list_transforms is None:
            self.list_transforms = [(StandardScaler, ),
                                    (QuantileTransformer, 30)]
        else:
            self.list_transforms = list_transforms

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

        contour = cat.reshape(-1, 1)
        transform = self.list_transforms[1]
        sc = transform[0]
        sc = sc(*transform[1:]).fit(contour)
        scalers.append(sc)

        return scalers

    def apply_transform(self, x, scaler):
        out = scaler.transform(x.reshape(-1, 1)).squeeze(-1)
        return out

    def get_quantized_loudness(self, e_l0, events):
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
        e_l0 = self.dataset["e_loudness"][idx:idx + self.n_sample]
        events = self.dataset["events"][idx:idx + self.n_sample]

        # Apply transforms :

        u_f0 = self.apply_transform(u_f0, self.scalers[0])
        e_f0 = self.apply_transform(e_f0, self.scalers[0])
        e_l0 = self.apply_transform(e_l0, self.scalers[1])

        u_f0 = torch.from_numpy(u_f0).float()
        e_f0 = torch.from_numpy(e_f0).float()
        e_l0 = torch.from_numpy(e_l0).float()
        events = torch.from_numpy(events).float()

        u_l0 = self.get_quantized_loudness(e_l0, events)

        # Change ranges from [0, 1] -> [-1, 1]

        u_f0 = 2 * (u_f0 - .5)
        u_l0 = 2 * (u_l0 - .5)
        e_f0 = 2 * (e_f0 - .5)
        e_l0 = 2 * (e_l0 - .5)

        model_input = torch.cat([
            u_f0.unsqueeze(-1),
            u_l0.unsqueeze(-1),
        ], -1)

        target = torch.cat([
            e_f0.unsqueeze(-1),
            e_l0.unsqueeze(-1),
        ], -1)

        return model_input, target
import torch

torch.set_grad_enabled(False)
from training import Network
#import matplotlib.pyplot as plt
import soundfile as sf

from pytorch_lightning.callbacks import ModelCheckpoint
from random import randint

from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from diffusion_dataset import DiffusionDataset
import numpy as np

import pickle


class DataLoader:
    def __init__(self, path, list_transforms):
        with open(path, "rb") as dataset:
            self.dataset = pickle.load(dataset)

        self.N = len(self.dataset["u_f0"])
        self.idx = 0
        self.n_sample = 2048
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

    def get_quantized_loudness(self, e_l0, onsets, offsets):
        e = torch.abs(onsets + offsets)
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

    def _get(self):
        i = self.idx * self.sample
        # update counter
        self.idx += 1

        i = min(i, len(self) * self.n_sample - self.n_sample)

        u_f0 = self.dataset["u_f0"][i:i + self.n_sample]
        e_f0 = self.dataset["e_f0"][i:i + self.n_sample]
        e_l0 = self.dataset["e_loudness"][i:i + self.n_sample]
        onsets = self.dataset["onsets"][i:i + self.n_sample]
        offsets = self.dataset["offsets"][i:i + self.n_sample]
        # Apply transforms :

        u_f0 = self.apply_transform(u_f0, self.scalers[0])
        e_f0 = self.apply_transform(e_f0, self.scalers[0])
        e_l0 = self.apply_transform(e_l0, self.scalers[1])

        u_f0 = torch.from_numpy(u_f0).float()
        e_f0 = torch.from_numpy(e_f0).float()
        e_l0 = torch.from_numpy(e_l0).float()
        onsets = torch.from_numpy(onsets).float()

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

        return model_input, cdt, onsets, offsets


list_transforms = [
    (MinMaxScaler, ),
    (QuantileTransformer, 30),
]

dataset = DiffusionDataset(list_transforms=list_transforms)
val_len = len(dataset) // 20
train_len = len(dataset) - val_len

train, val = random_split(dataset, [train_len, val_len])

# load data :

model = Network.load_from_checkpoint(
    "logs/diffusion/default/version_2/checkpoints/epoch=19686-step=157495.ckpt",
    strict=False).eval()

model.set_noise_schedule()
model.ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

N_EXAMPLE = 5
for i in range(N_EXAMPLE):
    _, midi = dataset[randint(0, len(dataset))]

    midi = midi.unsqueeze(0)

    n_step = 10
    out = model.sample(midi, midi)
    f0, lo = model.post_process(out)

    f0 = torch.from_numpy(f0).float().reshape(1, -1, 1)
    lo = torch.from_numpy(lo).float().reshape(1, -1, 1)

    audio = ddsp(f0, lo).reshape(-1).numpy()
    sf.write("results/diffusion/samples/sample{}.wav".format(i), audio, 16000)

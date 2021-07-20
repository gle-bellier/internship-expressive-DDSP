import torch

torch.set_grad_enabled(False)
from training import Network
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

    def inverse_transform(self, out):
        out = out / 2 + .5

        f0, l0 = torch.split(out, 1, -1)
        f0 = f0.reshape(-1, 1).cpu().numpy()
        l0 = l0.reshape(-1, 1).cpu().numpy()

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0).reshape(-1)
        l0 = self.scalers[1].inverse_transform(l0).reshape(-1)

        return f0, l0

    def get_quantized_loudness(self, e_l0, onsets, offsets):
        #compute sum of all events :
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

    def get(self):
        i = self.idx * self.n_sample
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

        return model_input, cdt, onsets, offsets


list_transforms = [
    (MinMaxScaler, ),
    (QuantileTransformer, 30),
]

PATH = "dataset/dataset-diffusion.pickle"
dataset = DataLoader(PATH, list_transforms=list_transforms)

model = Network.load_from_checkpoint(
    "logs/diffusion/default/version_2/checkpoints/epoch=19686-step=157495.ckpt",
    strict=False).eval()

model.set_noise_schedule()
model.ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

# Initialize data :

u_f0 = np.empty(0)
u_lo = np.empty(0)
e_f0 = np.empty(0)
e_lo = np.empty(0)
pred_f0 = np.empty(0)
pred_lo = np.empty(0)
onsets = np.empty(0)
offsets = np.empty(0)

# Prediction loops :

N_EXAMPLE = 5
for i in range(N_EXAMPLE):
    target, midi, ons, offs = dataset.get()

    n_step = 10
    out = model.sample(midi.unsqueeze(0), midi.unsqueeze(0))

    #f0, lo = model.post_process(out)
    f0, lo = dataset.inverse_transform(out)
    midi_f0, midi_lo = dataset.inverse_transform(midi)
    target_f0, target_lo = dataset.inverse_transform(target)

    # add to results:

    u_f0 = np.concatenate((u_f0, midi_f0))
    u_lo = np.concatenate((u_lo, midi_lo))

    e_f0 = np.concatenate((e_f0, target_f0))
    e_lo = np.concatenate((e_lo, target_lo))

    pred_f0 = np.concatenate((pred_f0, f0))
    pred_lo = np.concatenate((pred_lo, lo))

out = {
    "u_f0": u_f0,
    "u_lo": u_lo,
    "e_f0": e_f0,
    "e_lo": e_lo,
    "pred_f0": pred_f0,
    "pred_lo": pred_lo,
    "onsets": onsets,
    "offsets": offsets
}

with open("results/diffusion/data/results.pickle", "wb") as file_out:
    pickle.dump(out, file_out)

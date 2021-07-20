import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
import numpy as np

torch.set_grad_enabled(False)
from baseline_model import Model

from random import randint
import pickle


class DataLoader:
    def __init__(self, path, list_transforms):

        with open(path, "rb") as dataset:
            dataset = pickle.load(dataset)

        self.dataset = dataset
        self.N = len(dataset["u_f0"])
        self.list_transforms = list_transforms
        self.n_sample = 2050
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

        return model_input, target, onsets, offsets


list_transforms = [
    (MinMaxScaler, ),
    (QuantileTransformer, 30),
]

PATH = "dataset/dataset-diffusion.pickle"
dataset = DataLoader(PATH, list_transforms=list_transforms)

down_channels = [2, 16, 512, 1024]
ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

model = Model.load_from_checkpoint(
    "logs/baseline/default/version_1/checkpoints/epoch=114-step=1694.ckpt",
    scalers=dataset.scalers,
    channels=down_channels,
    ddsp=ddsp,
    strict=False).eval()

#model.ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

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
    model_input, target, ons, offs = dataset.get()

    n_step = 10
    out = model.generation_loop(model_input)

    f0, lo = dataset.inverse_transform(out)
    midi_f0, midi_lo = dataset.inverse_transform(midi)
    target_f0, target_lo = dataset.inverse_transform(target)

    # add to results:

    f0, lo = out.split(1, -1)
    midi_f0, midi_lo = midi.split(1, -1)
    target_f0, target_lo = target.split(1, -1)

    u_f0 = np.concatenate((u_f0, midi_f0.squeeze()))
    u_lo = np.concatenate((u_lo, midi_lo.squeeze()))

    e_f0 = np.concatenate((e_f0, target_f0.squeeze()))
    e_lo = np.concatenate((e_lo, target_lo.squeeze()))

    pred_f0 = np.concatenate((pred_f0, f0.squeeze()))
    pred_lo = np.concatenate((pred_lo, lo.squeeze()))

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

with open("results/unet-rnn/data/results-raw.pickle", "wb") as file_out:
    pickle.dump(out, file_out)

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

list_transforms = [
    (QuantileTransformer, 100),
    (QuantileTransformer, 30),
]
PATH = "dataset/dataset-diffusion.pickle"
dataset = DiffusionDataset(path=PATH,
                           list_transforms=list_transforms,
                           eval=True)

model = Network.load_from_checkpoint(
    "logs/diffusion/default/version_3/checkpoints/epoch=70510-step=564087.ckpt",
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
    target, midi, ons, offs = dataset[i]

    n_step = 10
    out = model.sample(midi.unsqueeze(0), midi.unsqueeze(0))

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

with open("results/diffusion/data/results2.pickle", "wb") as file_out:
    pickle.dump(out, file_out)

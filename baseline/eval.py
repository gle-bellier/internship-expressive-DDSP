import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
import numpy as np

torch.set_grad_enabled(False)
from baseline_model import Model
from baseline_dataset import Baseline_Dataset


from random import randint
import pickle


list_transforms = [
    (MinMaxScaler, ),  # pitch
    (QuantileTransformer, 120),  # lo
    (QuantileTransformer, 100),  # cents
]

PATH = "dataset/dataset-diffusion.pickle"
dataset = Baseline_Dataset(list_transforms=list_transforms, eval=True)

down_channels = [2, 16, 512, 1024]
ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

model = Model.load_from_checkpoint(
    "logs/baseline/default/version_0/checkpoints/epoch=547-step=2739.ckpt",
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
    model_input, target, ons, offs = dataset[i]



    n_step = 10
    out = model(model_input.unsqueeze(0))

    midi_p = model_input[..., :1]
    midi_cents = torch.zeros_like(midi_p)
    midi_lo = model_input[..., 1:2]

    e_cents = model_input[..., 2:3]
    e_lo = model_input[..., 3:4]

    pred_cents = out[..., :1]
    pred_lo = out[..., 1:2]

    f0, lo = dataset.post_processing(midi_p, pred_cents, pred_lo)
    midi_f0, midi_lo = dataset.post_processing(midi_p, midi_cents, midi_lo)
    target_f0, target_lo = dataset.post_processing(midi_p, e_cents, e_lo)

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

with open("results/baseline/data/results-raw.pickle", "wb") as file_out:
    pickle.dump(out, file_out)

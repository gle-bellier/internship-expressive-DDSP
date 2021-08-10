import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from unet_dataset import UNet_Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
import numpy as np

torch.set_grad_enabled(False)
from unet import UNet
from unet_dataset import UNet_Dataset

from random import randint
import pickle

list_transforms = [
    (MinMaxScaler, {}),
    (QuantileTransformer, {
        "n_quantiles": 30
    }),
]


def fit_transforms(u_f0, u_lo, list_transforms):
    scalers = []

    # pitch :

    contour = u_f0.reshape(-1, 1)
    transform = list_transforms[0]
    sc = transform[0]
    sc = sc(**transform[1]).fit(contour)
    scalers.append(sc)

    # loudness

    contour = self.dataset["e_loudness"]
    contour = contour.reshape(-1, 1)
    transform = self.list_transforms[1]
    sc = transform[0]
    sc = sc(**transform[1]).fit(contour)
    scalers.append(sc)

    return scalers


path = "dataset/test-set.pickle"
print("Loading Dataset...")
with open(path, "rb") as set:
    set = pickle.load(set)

PATH = "dataset/dataset-diffusion.pickle"
dataset = UNet_Dataset(PATH, list_transforms=list_transforms, eval=True)

down_channels = [2, 16, 512, 1024]
ddsp = torch.jit.load("ddsp_violin_pretrained.ts").eval()

model = UNet.load_from_checkpoint(
    "logs/unet/default/version_1/checkpoints/epoch=9074-step=136124.ckpt",
    scalers=dataset.scalers,
    channels=down_channels,
    ddsp=ddsp,
    strict=False).eval()

#model.ddsp = torch.jit.load("ddsp_violin_pretrained.ts").eval()

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
    midi, target, ons, offs = dataset[i]

    n_step = 10
    out = model(midi.unsqueeze(0))

    f0, lo = dataset.inverse_transform(out)
    midi_f0, midi_lo = dataset.inverse_transform(midi)
    target_f0, target_lo = dataset.inverse_transform(target)

    # # add to results:

    # f0, lo = out.split(1, -1)
    # midi_f0, midi_lo = midi.split(1, -1)
    # target_f0, target_lo = target.split(1, -1)

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

with open("results/unet-rnn/data/results.pickle", "wb") as file_out:
    pickle.dump(out, file_out)

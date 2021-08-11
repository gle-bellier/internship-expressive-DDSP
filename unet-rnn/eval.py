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

instrument = "violin"

dataset = UNet_Dataset(instrument=instrument,
                       list_transforms=list_transforms,
                       eval=True)

down_channels = [2, 16, 512, 1024]

model = UNet.load_from_checkpoint(
    "logs/unet-rnn/violin/default/version_0/checkpoints/epoch=3400-step=98628.ckpt",
    scalers=dataset.scalers,
    strict=False).eval()

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

with open("results/unet-rnn/data/results-{}.pickle".format(instrument),
          "wb") as file_out:
    pickle.dump(out, file_out)

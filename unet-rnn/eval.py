import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from unet_dataset import UNet_Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
import numpy as np

torch.set_grad_enabled(False)
from unet_rnn import UNet_RNN
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

train = UNet_Dataset(instrument=instrument,
                     type_set="train",
                     data_augmentation=True,
                     list_transforms=list_transforms)

dataset = UNet_Dataset(instrument=instrument,
                       list_transforms=list_transforms,
                       n_sample=1000,
                       scalers=train.scalers,
                       eval=True)

model = UNet_RNN.load_from_checkpoint(
    "logs/unet-rnn/violin/default/version_0/checkpoints/epoch=3400-step=98628.ckpt",
    scalers=train.scalers,
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

N_EXAMPLE = 20
for i in range(N_EXAMPLE):
    midi, target, ons, offs = dataset[i]

    out = model(midi.unsqueeze(0))

    f0, lo = dataset.inverse_transform(out)
    midi_f0, midi_lo = dataset.inverse_transform(midi)
    target_f0, target_lo = dataset.inverse_transform(target)

    # # add to results:

    # for raw results :

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

path = "results/unet-rnn/data/results-{}-testset.pickle".format(instrument)
print("Output file : ", path)

with open(path, "wb") as file_out:
    pickle.dump(out, file_out)

import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
import numpy as np

torch.set_grad_enabled(False)
from base import Baseline
from base_dataset import BaseDataset

from random import randint
import pickle

list_transforms = [
    (MinMaxScaler, {}),  # pitch
    (QuantileTransformer, {
        "n_quantiles": 120
    }),  # lo
    (QuantileTransformer, {
        "n_quantiles": 100
    }),  # cents
]

instrument = "violin"
CKPT = "logs/baseline2/violin/default/version_0/checkpoints/epoch=38-step=2222.ckpt"
print("Checkpoint path : ", CKPT)

train = BaseDataset(list_transforms=list_transforms,
                    type_set="train",
                    instrument=instrument,
                    eval=False)
dataset = BaseDataset(list_transforms=list_transforms,
                      type_set="test",
                      instrument=instrument,
                      scalers=train.scalers,
                      eval=True)

model = Baseline.load_from_checkpoint(CKPT,
                                      scalers=dataset.scalers,
                                      strict=False).eval()

model.ddsp = torch.jit.load("ddsp_{}_pretrained.ts".format(instrument)).eval()

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

    _, s_pred_cents, s_pred_lo = model.generation_loop(
        model_input.unsqueeze(0).float(), infer_pitch=True)

    s_u_p = model_input[:, :128]
    s_u_cents = torch.ones_like(model_input[:, :100]) * 0.5
    s_u_lo = model_input[:, 128:249]

    s_e_pitch = model_input[:, 251:379]
    s_e_cents = model_input[:, 379:479]
    s_e_lo = model_input[:, 479:600]

    s_pred_f0, s_pred_lo = dataset.post_processing(s_u_p, s_pred_cents,
                                                   s_pred_lo)
    s_u_f0, s_u_lo = dataset.post_processing(s_u_p, s_u_cents, s_u_lo)
    s_e_f0, s_e_lo = dataset.post_processing(s_u_p, s_e_cents, s_e_lo)

    u_f0 = np.concatenate((u_f0, s_u_f0.squeeze()))
    u_lo = np.concatenate((u_lo, s_u_lo.squeeze()))

    e_f0 = np.concatenate((e_f0, s_e_f0.squeeze()))
    e_lo = np.concatenate((e_lo, s_e_lo.squeeze()))

    pred_f0 = np.concatenate((pred_f0, s_pred_f0.squeeze()))
    pred_lo = np.concatenate((pred_lo, s_pred_lo.squeeze()))

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
out_file = "results/baseline/data/results-{}-midi.pickle".format(instrument)
print("Output file : ", out_file)
with open(out_file, "wb") as file_out:
    pickle.dump(out, file_out)

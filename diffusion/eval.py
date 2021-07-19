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

list_transforms = [
    (MinMaxScaler, ),
    (QuantileTransformer, 30),
]

dataset = DiffusionDataset(list_transforms=list_transforms)
val_len = len(dataset) // 20
train_len = len(dataset) - val_len

train, val = random_split(dataset, [train_len, val_len])

# down_channels = [2, 16, 256, 512, 1024]
# up_channels = [1024, 512, 256, 16, 2]
ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

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

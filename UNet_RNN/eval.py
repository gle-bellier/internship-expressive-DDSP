import torch

torch.set_grad_enabled(False)

#import matplotlib.pyplot as plt
import soundfile as sf
from UNet_RNN import UNet_RNN

from pytorch_lightning.callbacks import ModelCheckpoint
from random import randint

from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from UNet_dataset import UNet_Dataset
import numpy as np

import pickle

list_transforms = [
    (MinMaxScaler, ),
    (QuantileTransformer, 30),
]

dataset = UNet_Dataset(list_transforms=list_transforms)
val_len = len(dataset) // 20
train_len = len(dataset) - val_len
train, val = random_split(dataset, [train_len, val_len])

down_channels = [2, 16, 512, 1024]
ddsp = torch.jit.load("../ddsp_debug_pretrained.ts").eval()

model = UNet_RNN.load_from_checkpoint(
    "lightning_logs/version_7/checkpoints/epoch=9518-step=142784.ckpt",
    scalers=dataset.scalers,
    channels=down_channels,
    ddsp=ddsp,
    strict=False).eval()

NB_EXAMPLES = 5

for i in range(NB_EXAMPLES):
    model_input, target = dataset[randint(0, len(dataset))]

    model_input = model_input.unsqueeze(0)
    pred = model(model_input)
    f0, lo = model.post_process(pred)

    f0 = torch.from_numpy(f0).float().reshape(1, -1, 1)
    lo = torch.from_numpy(lo).float().reshape(1, -1, 1)

    audio = ddsp(f0, lo).reshape(-1).numpy()
    sf.write("sample{}.wav".format(i), audio, 16000)

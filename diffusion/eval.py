import torch

torch.set_grad_enabled(False)
from diffusion_model import UNet_Diffusion
#import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint

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

down_channels = [2, 16, 256, 512, 1024]
up_channels = [1024, 512, 256, 16, 2]
ddsp = torch.jit.load("../ddsp_debug_pretrained.ts").eval()

model = UNet_Diffusion.load_from_checkpoint(
    "lightning_logs/version_3/checkpoints/epoch=3739-step=56099.ckpt",
    scalers=dataset.scalers,
    down_channels=down_channels,
    up_channels=up_channels,
    ddsp=ddsp).eval()

model.set_noise_schedule()

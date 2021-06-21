import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
import pytorch_lightning as pl
from diffusion_dataset import DiffusionDataset
import matplotlib.pyplot as plt
from utils import *

if __name__ == "__main__":

    list_transforms = [
        (Identity, ),
        (QuantileTransformer, 30),
        (Identity, ),
        (QuantileTransformer, 30),
    ]

    dataset = DiffusionDataset()
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])
    for i in range(20):
        model_input, target = train[i]

        u_f0 = model_input[:, 0].squeeze()
        u_l0 = model_input[:, 1].squeeze()
        e_f0 = model_input[:, 2].squeeze()
        e_l0 = model_input[:, 3].squeeze()

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Dataset")

        ax1.plot(u_f0, label="Midi")
        ax1.plot(e_f0, label="Target")
        ax1.set_title("Frequency")
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax2.plot(u_l0, label="Midi")
        ax2.plot(e_l0, label="Target")
        ax2.set_title("Loudness")
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

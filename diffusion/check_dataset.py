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
        (QuantileTransformer, {"n_quantile":100, "output_distribution": "normal"}),
        (QuantileTransformer, {"n_quantile":30, "output_distribution": "normal"}),
    ]

    dataset = DiffusionDataset(list_transforms=list_transforms, n_sample=20048)

    for i in range(2):
        model_input, cdt = dataset[i]

        e_f0 = model_input[:, 0].squeeze()
        e_l0 = model_input[:, 1].squeeze()
        u_f0 = cdt[:, 0].squeeze()
        u_l0 = cdt[:, 1].squeeze()

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

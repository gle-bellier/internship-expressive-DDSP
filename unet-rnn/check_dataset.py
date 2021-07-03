import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
import pytorch_lightning as pl
from unet_dataset import UNet_Dataset
import matplotlib.pyplot as plt
from utils import *


def post_process(scalers, out):

    f0, l0 = torch.split(out, 1, -1)
    f0 = f0.reshape(-1, 1).cpu().numpy()
    l0 = l0.reshape(-1, 1).cpu().numpy()

    # Inverse transforms
    f0 = scalers[0].inverse_transform(f0).reshape(-1)
    l0 = scalers[1].inverse_transform(l0).reshape(-1)

    return f0, l0


if __name__ == "__main__":

    list_transforms = [
        (MinMaxScaler, ),
        (QuantileTransformer, 30),
    ]

    dataset = UNet_Dataset(list_transforms=list_transforms)
    scalers = dataset.scalers

    for i in range(100):
        model_input, target = dataset[i]

        # u_f0 = model_input[:, 0].squeeze()
        # u_l0 = model_input[:, 1].squeeze()
        # e_f0 = target[:, 0].squeeze()
        # e_l0 = target[:, 1].squeeze()

        # fig, (ax1, ax2) = plt.subplots(2)
        # fig.suptitle("Dataset")

        # ax1.plot(u_f0, label="Midi")
        # ax1.plot(e_f0, label="Target")
        # ax1.set_title("Frequency")
        # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # ax2.plot(u_l0, label="Midi")
        # ax2.plot(e_l0, label="Target")
        # ax2.set_title("Loudness")
        # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()

        # Post processing

        u_f0, u_l0 = post_process(scalers, model_input)
        e_f0, e_l0 = post_process(scalers, target)

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Dataset")

        ax1.plot(u_f0.squeeze(), label="Midi")
        ax1.plot(e_f0.squeeze(), label="Target")
        ax1.set_title("Frequency")
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax2.plot(u_l0.squeeze(), label="Midi")
        ax2.plot(e_l0.squeeze(), label="Target")
        ax2.set_title("Loudness")
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

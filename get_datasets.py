from tqdm import tqdm

import csv
import numpy as np
import matplotlib.pyplot as plt
import glob

from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer


def get_datasets(dataset_file="dataset/contours.csv",
                 sampling_rate=100,
                 sample_duration=20,
                 batch_size=16,
                 ratio=0.7,
                 pitch_transform=None,
                 loudness_transform=None):

    sample_length = sample_duration * sampling_rate

    u_f0 = []
    u_loudness = []
    e_f0 = []
    e_loudness = []
    e_f0_mean = []
    e_f0_stddev = []

    with open(dataset_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            u_f0.append(float(row["u_f0"]))
            u_loudness.append(float(row["u_loudness"]))
            e_f0.append(float(row["e_f0"]))
            e_loudness.append(float(row["e_loudness"]))
            e_f0_mean.append(float(row["e_f0_mean"]))
            e_f0_stddev.append(float(row["e_f0_stddev"]))

    u_f0 = np.array(u_f0)
    u_loudness = np.array(u_loudness)
    e_f0 = np.array(e_f0)
    e_loudness = np.array(e_loudness)
    e_f0_mean = np.array(e_f0_mean)
    e_f0_stddev = np.array(e_f0_stddev)

    fits = preprocessing(
        [u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev],
        pitch_transform, loudness_transform)

    full_length = len(u_f0)
    # we need to split the dataset between training and testing

    i_cut = int(ratio * full_length)

    train_u_f0 = u_f0[:i_cut]
    train_u_loudness = u_loudness[:i_cut]
    train_e_f0 = e_f0[:i_cut]
    train_e_loudness = e_loudness[:i_cut]
    train_e_f0_mean = e_f0_mean[:i_cut]
    train_e_f0_stddev = e_f0_stddev[:i_cut]

    train_length = len(train_u_f0)

    test_u_f0 = u_f0[i_cut:]
    test_u_loudness = u_loudness[i_cut:]
    test_e_f0 = e_f0[i_cut:]
    test_e_loudness = e_loudness[i_cut:]
    test_e_f0_mean = e_f0_mean[i_cut:]
    test_e_f0_stddev = e_f0_stddev[i_cut:]
    test_length = len(test_u_f0)

    print("Full size : {}s".format(full_length / sampling_rate))
    print("Training size : {}s".format(train_length / sampling_rate))
    print("Test size : {}s".format(test_length / sampling_rate))

    sc = MinMaxScaler()

    train_dataset = ContoursTrainDataset(train_u_f0,
                                         train_u_loudness,
                                         train_e_f0,
                                         train_e_loudness,
                                         train_e_f0_mean,
                                         train_e_f0_stddev,
                                         sample_length=sample_length)

    test_dataset = ContoursTestDataset(test_u_f0,
                                       test_u_loudness,
                                       test_e_f0,
                                       test_e_loudness,
                                       test_e_f0_mean,
                                       test_e_f0_stddev,
                                       sample_length=sample_length)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size)

    return train_loader, test_loader, fits


def preprocessing(data, pitch_transform, loudness_transform):
    list_fits = []
    print(len(data))
    for i in range(len(data)):

        contour = data[i].reshape(-1, 1)
        if i in [1, 3]:  # correspond to loudness indexes
            if loudness_transform == "Standardise":
                sc = StandardScaler()
                sc.fit(contour)
                list_fits.append(sc)

            elif loudness_transform == "Quantile":
                q = QuantileTransformer()
                q.fit(contour)
                list_fits.append(q)

            else:
                list_fits.append(None)

        else:
            if pitch_transform == "Standardise":
                sc = StandardScaler()
                sc.fit(contour)
                list_fits.append(sc)

            elif pitch_transform == "Quantile":
                q = QuantileTransformer()
                q.fit(contour)
                list_fits.append(q)

            else:
                list_fits.append(None)

    print(len(list_fits))
    return list_fits

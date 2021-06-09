import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ContoursTrainDataset(Dataset):
    """ Unexpressive and expressive contours Dataset"""
    def __init__(self,
                 u_f0,
                 u_loudness,
                 e_f0,
                 e_loudness,
                 e_f0_mean,
                 e_f0_stddev,
                 scalers,
                 sample_length=2000,
                 overlap=0.3):

        self.sample_length = sample_length
        self.overlap = overlap
        self.fits = scalers

        self.u_f0 = u_f0.reshape(-1, 1)
        self.u_loudness = u_loudness.reshape(-1, 1)
        self.e_f0 = e_f0.reshape(-1, 1)
        self.e_loudness = e_loudness.reshape(-1, 1)
        self.e_f0_mean = e_f0_mean.reshape(-1, 1)
        self.e_f0_stddev = e_f0_stddev.reshape(-1, 1)

        self.length = len(self.u_f0)
        self.segments = []

    def __len__(self):
        return int(self.length / ((1 - self.overlap) * self.sample_length))

    def get_random_indexes(self):

        seg_length = int((1 - self.overlap) * self.sample_length)
        i_max = np.floor((self.length - self.sample_length) / seg_length)

        i = np.random.randint(i_max + 1)
        return int(i * seg_length), int(i * seg_length + self.sample_length)

    def __getitem__(self, idx):
        start, end = self.get_random_indexes()
        #print("Indexes: [{}:{}]".format(start, end))
        self.segments.append((start, end))

        # get windows :

        u_f0, e_f0, e_f0_mean, e_f0_stddev = self.u_f0[start:end], self.e_f0[
            start:end], self.e_f0_mean[start:end], self.e_f0_stddev[start:end]
        u_loudness, e_loudness = self.u_loudness[start:end], self.e_loudness[
            start:end]

        # apply transforms :
        u_f0 = self.scalers[0].transform(u_f0)
        u_loudness = self.scalers[1].transform(u_loudness)
        e_f0 = self.scalers[2].transform(e_f0)
        e_loudness = self.scalers[3].transform(e_loudness)
        e_f0_mean = self.scalers[4].transform(e_f0_mean)
        e_f0_stddev = self.scalers[5].transform(e_f0_stddev)

        return u_f0.astype(float), u_loudness.astype(float), e_f0.astype(
            float), e_loudness.astype(float), e_f0_mean.astype(
                float), e_f0_stddev.astype(float)


class ContoursTestDataset(Dataset):
    """ Unexpressive and expressive contours Dataset"""
    def __init__(self,
                 u_f0,
                 u_loudness,
                 e_f0,
                 e_loudness,
                 e_f0_mean,
                 e_f0_stddev,
                 scalers,
                 sample_length=2000,
                 overlap=0.3):

        self.sample_length = sample_length
        self.overlap = overlap
        self.scalers = scalers

        self.u_f0 = u_f0.reshape(-1, 1)
        self.u_loudness = u_loudness.reshape(-1, 1)
        self.e_f0 = e_f0.reshape(-1, 1)
        self.e_loudness = e_loudness.reshape(-1, 1)
        self.e_f0_mean = e_f0_mean.reshape(-1, 1)
        self.e_f0_stddev = e_f0_stddev.reshape(-1, 1)

        self.length = len(self.u_f0)
        self.segments = []

    def __len__(self):
        return self.length // self.sample_length

    def __getitem__(self, idx):
        start = int(idx * self.sample_length)
        end = int(start + self.sample_length)
        self.segments.append((start, end))

        # get windows :

        u_f0, e_f0, e_f0_mean, e_f0_stddev = self.u_f0[start:end], self.e_f0[
            start:end], self.e_f0_mean[start:end], self.e_f0_stddev[start:end]
        u_loudness, e_loudness = self.u_loudness[start:end], self.e_loudness[
            start:end]

        # apply transforms :
        u_f0 = self.scalers[0].transform(u_f0)
        u_loudness = self.scalers[1].transform(u_loudness)
        e_f0 = self.scalers[2].transform(e_f0)
        e_loudness = self.scalers[3].transform(e_loudness)
        e_f0_mean = self.scalers[4].transform(e_f0_mean)
        e_f0_stddev = self.scalers[5].transform(e_f0_stddev)

        return u_f0.astype(float), u_loudness.astype(float), e_f0.astype(
            float), e_loudness.astype(float), e_f0_mean.astype(
                float), e_f0_stddev.astype(float)

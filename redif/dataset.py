import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_step):
        super().__init__()
        self.u_f0 = torch.from_numpy(np.load("u_f0.npy")).long()
        self.u_lo = torch.from_numpy(np.load("u_lo.npy")).float()
        self.e_f0 = torch.from_numpy(np.load("e_f0.npy")).float()
        self.e_lo = torch.from_numpy(np.load("e_lo.npy")).float()

        self.n_step = n_step

    def __len__(self):
        return len(self.u_f0) // self.n_step

    def __getitem__(self, idx):
        u_f0 = self.u_f0[idx * self.n_step:(idx + 1) * self.n_step]
        u_lo = self.u_lo[idx * self.n_step:(idx + 1) * self.n_step]
        e_f0 = self.e_f0[idx * self.n_step:(idx + 1) * self.n_step]
        e_lo = self.e_lo[idx * self.n_step:(idx + 1) * self.n_step]
        return u_f0, u_lo, e_f0, e_lo

import torch
import numpy as np
from os import path


def load(data_path, name, dtype=torch.long):
    """
    loads a numpy array located in data_path/name and convert it
    to a torch.Tensor with dtype 
    """
    array = np.load(path.join(data_path, name))
    tensor = torch.from_numpy(array)
    tensor = tensor.to(dtype)
    return tensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_step, data_path):
        super().__init__()
        self.u_f0 = load(data_path, "u_f0.npy", torch.long)
        self.u_lo = load(data_path, "u_lo.npy", torch.float)
        self.e_f0 = load(data_path, "e_f0.npy", torch.float)
        self.e_lo = load(data_path, "e_lo.npy", torch.float)

        self.n_step = n_step

    def __len__(self):
        return len(self.u_f0) // self.n_step

    def __getitem__(self, idx):
        u_f0 = self.u_f0[idx * self.n_step:(idx + 1) * self.n_step]
        u_lo = self.u_lo[idx * self.n_step:(idx + 1) * self.n_step]
        e_f0 = self.e_f0[idx * self.n_step:(idx + 1) * self.n_step]
        e_lo = self.e_lo[idx * self.n_step:(idx + 1) * self.n_step]
        return u_f0, u_lo, e_f0, e_lo

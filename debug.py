#%%
import torch
import numpy as np
from redif.model import Model
from redif.dataset import Dataset

dataset = Dataset(2048)
model = Model(2, 128, [128, 256, 256, 512])

model.transform.compute_stats(dataset.e_f0, dataset.e_lo)

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



class ContoursDataset(Dataset):
    """ Unexpressive and expressive contours Dataset"""


    def __init__(self, csv_file, transform = None):

        self.contours = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.contours)

    def __getitem__(self):
        pass




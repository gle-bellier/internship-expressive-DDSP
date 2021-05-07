from __future__ import print_function, division
import os
import torch
import pandas as pd
from tqdm import tqdm

import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob
from get_contours import ContoursGetter


class ContoursDataset(Dataset):
    """ Unexpressive and expressive contours Dataset"""


    def __init__(self, dataset_path, transform = None):

        self.transform = transform

        self.dataset_path = dataset_path
        filenames =[file[len(dataset_path):-4] for file in glob.glob(self.dataset_path + "*.mid")]

        self.u_f0 = np.empty(0)
        self.u_loudness = np.empty(0)
        self.e_f0 = np.empty(0)
        self.e_loudness = np.empty(0)
        


        with tqdm(total=len(filenames)) as pbar:
            for filename in filenames:
                midi_file = filename + ".mid"
                wav_file = filename + ".wav"
                g = ContoursGetter()
                u_f0, u_loudness, e_f0, e_loudness = g.get_contours(dataset_path, midi_file, wav_file, sampling_rate=16000, block_size=160, max_silence_duration=3, verbose=False)
                pbar.update(1)



        # load all contours : 


    def __len__(self):
        return len(self.register)



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pass



if __name__ == "__main__":


    dataset_path = "dataset-midi-wav/"
    g = ContoursDataset(dataset_path)
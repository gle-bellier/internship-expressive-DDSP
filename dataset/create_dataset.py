from __future__ import print_function, division
import os
import torch
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



class ContoursDataset(Dataset):
    """ Unexpressive and expressive contours Dataset"""


    def __init__(self, register_csv_file, files_folder, transform = None):

        self.register = pd.read_csv(register_csv_file+".csv")
        self.transform = transform
        self.files_folder = files_folder + "/"


        # load all contours : 

        self.lists_u_f0, self.lists_u_loudness, self.lists_e_f0, self.lists_e_loudness = self.load_data() 

    def __len__(self):
        return len(self.register)


    def load_data(self):

        lists_u_f0 = []
        lists_u_loudness = []
        lists_e_f0 = []
        lists_e_loudness = []
        
        for i in range(len(self.register)):
            row = self.register.iloc[i]
            filename = row["file_name"]
            instrument = row["instrument"]
            print("Filename : {}\n Instrument : {}".format(filename, instrument))



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pass


    def get_contours_from_csv(self, filename):
        u_f0 = []
        u_loudness = []
        e_f0 = []
        e_loudness = []

        with open(self.files_folder + filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                u_f0.append(float(row["u_f0"]))
                u_loudness.append(float(row["u_loudness"]))
                e_f0.append(float(row["e_f0"]))
                e_loudness.append(float(row["e_loudness"]))
                
        return np.array(u_f0), np.array(u_loudness), np.array(e_f0), np.array(e_loudness)



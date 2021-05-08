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
                u_f0_track, u_loudness_track, e_f0_track, e_loudness_track = g.get_contours(dataset_path, midi_file, wav_file, sampling_rate=16000, block_size=160, max_silence_duration=3, verbose=False)
                
                self.u_f0 = np.concatenate((self.u_f0, u_f0_track))
                self.u_loudness = np.concatenate((self.u_loudness, u_loudness_track))
                self.e_f0 = np.concatenate((self.e_f0, e_f0_track))
                self.e_loudness = np.concatenate((self.e_loudness, e_loudness_track))

                pbar.update(1)
        
        self.length = len(self.u_f0)
        self.segments = []



    def __len__(self):
        return self.length


    def get_random_indexes(self, length = 2000, overlap = 0.0):

        seg_length = int((1 - overlap) * length)
        i_max = np.floor((self.length - length)/seg_length)

        i = np.random.randint(i_max+1)
        return i*seg_length, i*seg_length + length






    def __getitem__(self, idx):
        
        start, end = self.get_random_indexes(length=2000, overlap=0.5)
        #print("Indexes: [{}:{}]".format(start, end))
        self.segments.append((start,end))

        return self.u_f0[start:end], self.u_loudness[start:end], self.e_f0[start:end], self.e_loudness[start:end]

        
        




if __name__ == "__main__":

    VERBOSE = False
    dataset_path = "dataset-midi-wav/"
    g = ContoursDataset(dataset_path)
    print("Full size : {}".format(g.length/100))

    for i in range(200):
        u_f0, u_loudness, e_f0, e_loudness = g[0]
        
        if VERBOSE:
            plt.plot(e_f0, label = "wav")
            plt.plot(u_f0, label = "midi" )
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title("Frequency comparison")
            plt.show()


            plt.plot(e_loudness, label = "wav")
            plt.plot(u_loudness, label = "midi" )
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title("Loudness comparison")
            plt.show()


    if VERBOSE:
        n = g.length
        for i in range(len(g.segments)):
            v = np.zeros(n)
            a, b = g.segments[i]
            v[a:b] = 0.5 + i/1000
            plt.plot(v)
        plt.title("Segments")
        plt.show()
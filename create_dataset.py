from __future__ import print_function, division
import os
import torch
import pandas as pd
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob


from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset

sampling_rate = 100
dataset_path = "dataset-midi-wav/"
filenames =[file[len(dataset_path):-4] for file in glob.glob(dataset_path + "*.mid")]


u_f0 = np.empty(0)
u_loudness = np.empty(0)
e_f0 = np.empty(0)
e_loudness = np.empty(0)


list_indexes = [] # store the indexes of ends of tracks
index_end = 0


with tqdm(total=len(filenames)) as pbar:
    for filename in filenames:
        midi_file = filename + ".mid"
        wav_file = filename + ".wav"
        g = ContoursGetter()
        u_f0_track, u_loudness_track, e_f0_track, e_loudness_track = g.get_contours(dataset_path, midi_file, wav_file, sampling_rate=16000, block_size=160, max_silence_duration=3, verbose=False)
        
        u_f0 = np.concatenate((u_f0, u_f0_track))
        u_loudness = np.concatenate((u_loudness, u_loudness_track))
        e_f0 = np.concatenate((e_f0, e_f0_track))
        e_loudness = np.concatenate((e_loudness, e_loudness_track))


        index_end +=  len(u_f0_track)
        list_indexes.append(index_end)

        pbar.update(1)


full_length = len(u_f0)
# we need to split the dataset between training and testing


ratio = 0.7
i_cut = np.floor(ratio * full_length)

i = 0 
while i<len(list_indexes) and list_indexes[i]< i_cut:
    i+=1
i_cut_real = list_indexes[i]

train_u_f0 = u_f0[:i_cut_real]
train_u_loudness = u_loudness[:i_cut_real]
train_e_f0 = u_f0[:i_cut_real]
train_e_loudness = u_loudness[:i_cut_real]
train_length = len(train_u_f0)    

test_u_f0 = u_f0[i_cut_real:]
test_u_loudness = u_loudness[i_cut_real:]
test_e_f0 = u_f0[i_cut_real:]
test_e_loudness = u_loudness[i_cut_real:]
test_length = len(test_u_f0)    




print("Full size : {}s".format(full_length/sampling_rate))
print("Training size : {}s".format(train_length/sampling_rate))
print("Test size : {}s".format(test_length/sampling_rate))




train_dataset = ContoursTrainDataset(train_u_f0, train_u_loudness, train_e_f0, train_e_loudness, sample_length=2000, transform=None)
test_dataset = ContoursTestDataset(test_u_f0, test_u_loudness, test_e_f0, test_e_loudness, sample_length=2000, transform=None)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12)


print("Size train set : ", len(train_dataset))
print("Size test set : ", len(test_dataset))


# for train_sample in train_loader:
#     print(train_sample[0].shape)

# for test_sample in test_loader:
#     print(test_sample[0].shape)










VERBOSE = True

if VERBOSE:
    for i in range(len(train_dataset.segments)):
        v = np.zeros(full_length)
        a, b = train_dataset.segments[i]
        v[a:b] = 0.5 + i/1000
        plt.plot(v)
    plt.title("Segments")
    plt.show()
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import glob


from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.LSTMContours import LSTMContours

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler





def get_datasets(dataset_path = "dataset-midi-wav/", sampling_rate = 100, sample_duration = 20, batch_size = 16, ratio = 0.7, transform=None):
    
    
    filenames =[file[len(dataset_path):-4] for file in glob.glob(dataset_path + "*.mid")]
     # ration between train and test datasets
        
    sample_length = sample_duration*sampling_rate 


    u_f0 = np.empty(0)
    u_loudness = np.empty(0)
    e_f0 = np.empty(0)
    e_loudness = np.empty(0)
    e_f0_mean = np.empty(0)
    e_f0_stddev = np.empty(0)

    list_indexes = [] # store the indexes of ends of tracks
    index_end = 0


    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            midi_file = filename + ".mid"
            wav_file = filename + ".wav"
            g = ContoursGetter()
            u_f0_track, u_loudness_track, e_f0_track, e_loudness_track, e_f0_mean_track, e_f0_stddev_track = g.get_contours(dataset_path, midi_file, wav_file, sampling_rate=16000, block_size=160, max_silence_duration=3, verbose=False)
            
            u_f0 = np.concatenate((u_f0, u_f0_track))
            u_loudness = np.concatenate((u_loudness, u_loudness_track))
            e_f0 = np.concatenate((e_f0, e_f0_track))
            e_loudness = np.concatenate((e_loudness, e_loudness_track))

            e_f0_mean = np.concatenate((e_f0_mean, e_f0_mean_track))
            e_f0_stddev = np.concatenate((e_f0_stddev, e_f0_stddev_track))

            index_end +=  len(u_f0_track)
            list_indexes.append(index_end)

            pbar.update(1)


    full_length = len(u_f0)
    # we need to split the dataset between training and testing



    i_cut = np.floor(ratio * full_length)

    i = 0 
    while i<len(list_indexes) and list_indexes[i]< i_cut:
        i+=1
    i_cut_real = list_indexes[i]

    train_u_f0 = u_f0[:i_cut_real]
    train_u_loudness = u_loudness[:i_cut_real]
    train_e_f0 = e_f0[:i_cut_real]
    train_e_loudness = e_loudness[:i_cut_real]
    train_e_f0_mean = e_f0_mean[:i_cut_real]
    train_e_f0_stddev = e_f0_stddev[:i_cut_real]

    train_length = len(train_u_f0)    

    test_u_f0 = u_f0[i_cut_real:]
    test_u_loudness = u_loudness[i_cut_real:]
    test_e_f0 = e_f0[i_cut_real:]
    test_e_loudness = u_loudness[i_cut_real:]
    test_e_f0_mean = e_f0_mean[i_cut_real:]
    test_e_f0_stddev = e_f0_stddev[i_cut_real:]
    test_length = len(test_u_f0)    




    print("Full size : {}s".format(full_length/sampling_rate))
    print("Training size : {}s".format(train_length/sampling_rate))
    print("Test size : {}s".format(test_length/sampling_rate))


    sc = MinMaxScaler()

    train_dataset = ContoursTrainDataset(train_u_f0, train_u_loudness, train_e_f0, train_e_loudness, train_e_f0_mean, train_e_f0_stddev, sample_length=sample_length, transform=transform)
    test_dataset = ContoursTestDataset(test_u_f0, test_u_loudness, test_e_f0, test_e_loudness, test_e_f0_mean, test_e_f0_stddev, sample_length=sample_length, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
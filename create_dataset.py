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







##### CREATE TRAIN AND TEST DATASET #####


sampling_rate = 100
dataset_path = "dataset-midi-wav/"
filenames =[file[len(dataset_path):-4] for file in glob.glob(dataset_path + "*.mid")]
ratio = 0.7 # ration between train and test datasets
batch_size = 16


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





train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


print("train set : {} batches".format(len(train_dataset)))
print("test set : {} batches".format(len(test_dataset)))


for train_sample in train_loader:
    pass

for test_sample in test_loader:
    pass




### PLOT SAMPLES USED FOR TRAINING AND TESTING ###
VERBOSE = False
if VERBOSE:
    for i in range(len(train_dataset.segments)):
        v = np.zeros(full_length)
        a, b = train_dataset.segments[i]
        v[a:b] = 0.5 + i/1000
        plt.plot(v, color = "blue")

    for i in range(len(test_dataset.segments)):
        v = np.zeros(full_length)
        a, b = test_dataset.segments[i]
        v[i_cut_real + a:i_cut_real + b] = 0.5 + i/1000
        plt.plot(v, color = "red")

    plt.title("Segments")
    plt.show()    

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)

model = LSTMContours().to(device)
print(model.parameters)
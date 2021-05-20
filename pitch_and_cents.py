from tqdm import tqdm
from time import time
import librosa as li
import numpy as np
import matplotlib.pyplot as plt
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

train_loader, test_loader = get_datasets(dataset_file = "dataset/contours.csv", sampling_rate = 100, sample_duration = 20, batch_size = 1, ratio = 0.7, transform = None)

print("Size Train set : ", len(train_loader))
def frequencies_to_pitch_cents(frequencies, pitch_size):
    
    # one hot vectors : 
    pitch_array = torch.zeros(frequencies.size(0), frequencies.size(1))
    cents_array = torch.zeros(frequencies.size(0), frequencies.size(1))
    

    midi_pitch = torch.tensor(li.hz_to_midi(frequencies))
    midi_pitch = torch.round(midi_pitch).long()



    #print("Min =  {};  Max =  {} frequencies".format(li.midi_to_hz(0), li.midi_to_hz(pitch_size-1)))
    midi_pitch_clip = torch.clip(midi_pitch, min = 0, max = pitch_size-1)
    round_freq = torch.tensor(li.midi_to_hz(midi_pitch))
    
    cents = (1200 * torch.log2(frequencies / round_freq)).long()


    for i in range(0, pitch_array.size(0)):
        for j in range(0, pitch_array.size(1)):
            pitch_array[i, j] = midi_pitch_clip[i, j, 0]

    for i in range(0, cents_array.size(0)):
        for j in range(0, cents_array.size(1)):
            cents_array[i, j] = cents[i, j, 0] + 50


    return pitch_array, cents_array, midi_pitch_clip, cents


def pitch_cents_to_frequencies(pitch, cents):

    gen_pitch = pitch
    gen_cents = cents - 50

    gen_freq = torch.tensor(li.midi_to_hz(gen_pitch)) * torch.pow(2, gen_cents/1200)
    gen_freq = torch.unsqueeze(gen_freq, -1)

    return gen_freq



data = iter(train_loader)

for i in range(3):

    u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = next(data)
    u_f0 = torch.Tensor(u_f0.float())

    pitch_size, cents_size = 100, 100
    convert_pitch, convert_cents, midi_pitch_clip, cents = frequencies_to_pitch_cents(e_f0, pitch_size)

    print("Convert size : ", convert_pitch.size())
    convert_back_f0 = pitch_cents_to_frequencies(convert_pitch, convert_cents)


    e_f0_array = e_f0.squeeze().detach()
    convert_f0_array = convert_back_f0.squeeze().detach()


    plt.plot(e_f0_array, label = "Origin")
    plt.plot(convert_f0_array, label = "Convert")
    plt.legend()
    plt.show()



    plt.plot(midi_pitch_clip.squeeze().detach()/127, label = "Pitch")
    plt.plot(cents.squeeze().detach()/50, label = "Cents")
    plt.legend()
    plt.show()


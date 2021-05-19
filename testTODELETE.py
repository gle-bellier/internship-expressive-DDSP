from tqdm import tqdm
from time import time
import librosa as li
import numpy as np
import matplotlib.pyplot as plt
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.LSTMwithBCE import LSTMContoursBCE


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

train_loader, test_loader = get_datasets(dataset_file = "dataset/contours.csv", sampling_rate = 100, sample_duration = 20, batch_size = 1, ratio = 0.7, transform = None)

print(len(train_loader))
def frequencies_to_pitch_cents(frequencies, pitch_size, cents_size):
    
    # one hot vectors : 
    pitch_array = torch.zeros(frequencies.size(0), frequencies.size(1), pitch_size)
    cents_array = torch.zeros(frequencies.size(0), frequencies.size(1), cents_size)
    
    min_freq = li.midi_to_hz(0)
    max_freq = li.midi_to_hz(pitch_size-1)

    print("Min {} Max {} frequencies".format(min_freq, max_freq))
    frequencies = torch.clip(frequencies, min = min_freq, max = max_freq)

    midi_pitch = torch.tensor(li.hz_to_midi(frequencies))
    midi_pitch = torch.round(midi_pitch).long()


    round_freq = torch.tensor(li.midi_to_hz(midi_pitch))
    cents = (1200 * torch.log2(frequencies / round_freq)).long()


    for i in range(0, pitch_array.size(0)):
        for j in range(0, pitch_array.size(1)):
            pitch_array[i, j, midi_pitch[i, j, 0]] = 1

    for i in range(0, pitch_array.size(0)):
        for j in range(0, pitch_array.size(1)):
            pitch_array[i, j, cents[i, j, 0] + 50] = 1


    return pitch_array, cents_array


def pitch_cents_to_frequencies(pitch, cents):

    gen_pitch = torch.argmax(pitch, dim = -1)
    gen_cents = torch.argmax(cents, dim = -1) - 50

    gen_freq = torch.tensor(li.midi_to_hz(gen_pitch)) * torch.pow(2, gen_cents/1200)

    gen_freq = torch.unsqueeze(gen_freq, -1)

    return gen_freq






input_size = 32
hidden_size = 64

data = iter(train_loader)

for i in range(10):


    u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = next(data)

    u_f0 = torch.Tensor(u_f0.float())

    pitch_size, cents_size = 100, 100
    convert_pitch, convert_cents = frequencies_to_pitch_cents(u_f0, pitch_size, cents_size)
    convert_back_f0 = pitch_cents_to_frequencies(convert_pitch, convert_cents)


    u_f0_array = u_f0.squeeze().detach()
    convert_f0_array = convert_back_f0.squeeze().detach()


    plt.plot(u_f0_array, label = "Origin")
    plt.plot(convert_f0_array, label = "Convert")

    plt.legend()
    plt.show()



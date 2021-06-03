import io as io
import scipy.io.wavfile as wav

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

from sklearn.preprocessing import StandardScaler

from get_datasets import get_datasets
from models.benchmark_models import *
from models.LSTM_towards_realistic_midi import LSTMContours


def std_transform(v):
    std = torch.std(v, dim=1, keepdim=True)
    m = torch.mean(v, dim=1, keepdim=True)

    return (v - m) / std, m, std


def std_inv_transform(v, m, std):
    return v * std + m


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)

save_path = "results/saved_models/"
model_name = "LSTM_towards_realistic_midi3540epochs.pt"
wav_path = "results/saved_samples/"

model = LSTMContours().to(device)
model.load_state_dict(
    torch.load(save_path + model_name, map_location=torch.device("cpu")))
model.eval()

PATH = save_path + model_name
print(model.parameters)

sampling_rate = 100
number_of_examples = 5
RESYNTH = True

_, test_loader = get_datasets(dataset_file="dataset/contours.csv",
                              sampling_rate=sampling_rate,
                              sample_duration=20,
                              batch_size=1,
                              ratio=0.7,
                              transform=None)  #sc.fit_transform)
test_data = iter(test_loader)

ddsp = torch.jit.load("results/ddsp_debug_pretrained.ts")

model.eval()
with torch.no_grad():
    for i in range(number_of_examples):
        print("Sample {} reconstruction".format(i))

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = next(
            test_data)

        u_f0_norm, u_f0_mean, u_f0_std = std_transform(u_f0[:, 1:])
        u_loudness_norm, u_loudness_mean, u_loudness_std = std_transform(
            u_loudness[:, 1:])

        e_f0_norm, e_f0_mean, e_f0_std = std_transform(e_f0[:, 1:])
        e_loudness_norm, e_loudness_mean, e_loudness_std = std_transform(
            e_loudness[:, 1:])

        u_f0_norm = u_f0_norm.float()
        u_loudness_norm = u_loudness_norm.float()

        out_f0, out_loudness = model.predict(u_f0_norm, u_loudness_norm)

        out_f0 = std_inv_transform(out_f0, e_f0_mean, e_f0_std).float()
        out_loudness = std_inv_transform(out_loudness, e_loudness_mean,
                                         e_loudness_std).float()

        plt.plot(u_f0.squeeze(), label="midi")
        plt.plot(e_f0.squeeze(), label="perf")
        plt.plot(out_f0.squeeze(), label="model")
        plt.legend()
        plt.show()

        plt.plot(u_loudness.squeeze(), label="midi")
        plt.plot(e_loudness.squeeze(), label="perf")
        plt.plot(out_loudness.squeeze(), label="model")
        plt.legend()
        plt.show()

        model_audio = ddsp(out_f0, out_loudness).detach().squeeze().numpy()
        filename = "{}{}-sample{}.wav".format(wav_path, model_name[:-3], i)
        write(filename, 16000, model_audio)

        if RESYNTH:
            resynth_audio = ddsp(
                e_f0.float(), e_loudness.float()).detach().squeeze().numpy()
            filename = "{}{}-sample{}-resynth.wav".format(
                wav_path, model_name[:-3], i)
            write(filename, 16000, model_audio)

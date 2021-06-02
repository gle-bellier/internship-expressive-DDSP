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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)

save_path = "results/saved_models/"
model_name = "LSTM_towards_realistic_midi6613epochs.pt"
wav_path = "results/saved_samples/"

model = LSTMContours().to(device)
model.load_state_dict(
    torch.load(save_path + model_name, map_location=torch.device("cpu")))
model.eval()

print(model.parameters)

PATH = save_path + model_name
model = LSTMContours()
print(model.parameters)

sc_pitch = StandardScaler()
sc_loudness = StandardScaler()

sampling_rate = 100
number_of_examples = 1
RESYNTH = False

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

        plt.plot(u_f0.squeeze(), label="midi")
        plt.plot(e_f0.squeeze(), label="perf")
        plt.show()

        plt.plot(u_loudness.squeeze(), label="midi")
        plt.plot(e_loudness.squeeze(), label="perf")
        plt.show()

        u_f0_norm = torch.squeeze(u_f0[:, 1:], 0)
        u_f0_norm = torch.tensor(sc_pitch.fit_transform(u_f0_norm)).float()
        u_f0_norm = torch.unsqueeze(u_f0_norm, 0)

        u_loudness_norm = torch.squeeze(u_loudness[:, 1:], 0)
        u_loudness_norm = torch.tensor(
            sc_loudness.fit_transform(u_loudness_norm)).float()
        u_loudness_norm = torch.unsqueeze(u_loudness_norm, 0)

        print(u_f0_norm.shape)

        plt.plot(u_f0_norm.squeeze(), label="midi")
        plt.legend()
        plt.show()

        plt.plot(u_loudness_norm.squeeze(), label="midi")
        plt.legend()
        plt.show()

        out_f0, out_loudness = model.predict(u_f0_norm, u_loudness_norm)

        out_f0 = torch.squeeze(out_f0, 0)
        out_f0 = torch.tensor(sc_pitch.inverse_transform(out_f0))

        out_loudness = torch.squeeze(out_loudness, 0)
        out_loudness = torch.tensor(
            sc_loudness.inverse_transform(out_loudness))

        plt.plot(u_f0.squeeze(), label="midi")
        plt.plot(out_f0.squeeze(), label="model")
        plt.legend()
        plt.show()

        plt.plot(u_loudness.squeeze(), label="midi")
        plt.plot(out_loudness.squeeze(), label="model")
        plt.legend()
        plt.show()

        out_f0 = out_f0.unsqueeze(0).unsqueeze(-1)
        out_loudness = out_loudness.unsqueeze(0).unsqueeze(-1)

        model_audio = ddsp(out_f0, out_loudness).detach().squeeze().numpy()
        filename = "{}{}-sample{}.wav".format(wav_path, model_name[:-3], i)
        write(filename, 16000, model_audio)

        if RESYNTH:
            resynth_audio = ddsp(e_f0, e_loudness).detach().squeeze().numpy()
            filename = "{}{}-sample{}-resynth.wav".format(
                wav_path, model_name[:-3], i)
            write(filename, 16000, model_audio)

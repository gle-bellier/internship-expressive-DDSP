import io as io
import scipy.io.wavfile as wav

import torch

torch.set_grad_enabled(False)
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
from evaluation import Evaluator

from models.benchmark_models import *
from models.LSTM_towards_realistic_midi import LSTMContours

from os import makedirs


def std_transform(v):
    std = torch.std(v, dim=1, keepdim=True)
    m = torch.mean(v, dim=1, keepdim=True)

    return (v - m) / std, m, std


def std_inv_transform(v, m, std):
    return v * std + m


device = torch.device("cpu")
print('using', device)

save_path = "results/saved_models/"
model_name = "LSTM_towards_realistic_midi3540epochs.pt"
wav_path = "results/saved_samples/"

makedirs(wav_path, exist_ok=True)

model = LSTMContours().to(device)
model.load_state_dict(
    torch.load(save_path + model_name, map_location=torch.device("cpu")))
model.eval()

PATH = save_path + model_name
print(model.parameters)

sampling_rate = 100
number_of_examples = 5
RESYNTH = True

_, test_loader, fits = get_datasets(
    dataset_file="dataset/contours.csv",
    sampling_rate=sampling_rate,
    sample_duration=20,
    batch_size=1,
    ratio=0.7,
    pitch_transform="Quantile",
    loudness_transform="Standardise")  #sc.fit_transform)
test_data = iter(test_loader)

u_f0_fit, u_loudness_fit, e_f0_fit, e_loudness_fit, e_f0_mean_fit, e_f0_std_fit = fits

ddsp = torch.jit.load("results/ddsp_debug_pretrained.ts")
ddsp.eval()

with torch.no_grad():
    for i in range(number_of_examples):
        print("Sample {} reconstruction".format(i))

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = next(
            test_data)

        target = torch.cat([e_f0[:, 1:], e_loudness[:, 1:]], -1)

        u_f0_norm, u_f0_mean, u_f0_std = std_transform(u_f0[:, 1:])
        u_loudness_norm, u_loudness_mean, u_loudness_std = std_transform(
            u_loudness[:, 1:])

        u_f0_norm = torch.tensor(u_f0_fit.transform(
            u_f0[:, 1:].squeeze(0))).unsqueeze(0)
        u_loudness_norm = torch.tensor(
            u_loudness_fit.transform(u_loudness[:,
                                                1:].squeeze(0))).unsqueeze(0)

        e_f0_norm = torch.tensor(e_f0_fit.transform(
            e_f0[:, 1:].squeeze(0))).unsqueeze(0)
        e_loudness_norm = torch.tensor(
            e_loudness_fit.transform(e_loudness[:,
                                                1:].squeeze(0))).unsqueeze(0)

        u_f0_norm = u_f0_norm.float()
        u_loudness_norm = u_loudness_norm.float()

        out_f0, out_loudness = model.predict(u_f0_norm, u_loudness_norm)

        out_f0 = torch.tensor(u_f0_fit.inverse_transform(
            out_f0.squeeze(0))).unsqueeze(0)
        out_loudness = torch.tensor(
            u_loudness_fit.inverse_transform(
                out_loudness.squeeze(0))).unsqueeze(0)

        model_out = torch.cat([out_f0, out_loudness], -1)

        e = Evaluator()
        score = e.evaluate(model_out, target, PLOT=True)
        out_audio = e.listen(model_out, target, ddsp,
                             wav_path + model_name + "-{}.wav".format(i))

        print("Score : ", score)

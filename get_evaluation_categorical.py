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
from utils import *

from os import makedirs


def inv_transform(data, transforms):
    transform_data = []
    for i in range(len(data)):
        transform_data.append(
            torch.tensor(transforms[i].inverse_transform(
                data[i].squeeze(0))).unsqueeze(0))
    return transform_data


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

_, test_loader, fits = get_datasets(dataset_file="dataset/contours.csv",
                                    sampling_rate=sampling_rate,
                                    sample_duration=20,
                                    batch_size=1,
                                    ratio=0.7,
                                    pitch_transform="Quantile",
                                    loudness_transform="Quantile",
                                    pitch_n_quantiles=100,
                                    loudness_n_quantiles=100)

test_data = iter(test_loader)

u_f0_fit, u_loudness_fit, e_f0_fit, e_loudness_fit, e_f0_mean_fit, e_f0_std_fit = fits

ddsp = torch.jit.load("results/ddsp_debug_pretrained.ts")
ddsp.eval()

with torch.no_grad():
    for i in range(number_of_examples):
        print("Sample {} reconstruction".format(i))

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_dev = next(
            test_data)

        # CONTINUOUS TO CATEGORICAL
        u_pitch_cat = get_data_categorical(u_f0, n_out=100)
        u_loudness_cat = get_data_categorical(u_loudness, n_out=100)

        u_pitch_cat = u_pitch_cat[:, 1:].float()
        u_loudness_cat = u_loudness_cat[:, 1:].float()

        # PREDICTION
        out_f0, out_loudness = model.predict(u_f0, u_loudness)

        out_f0, out_loudness = inv_transform([out_f0, out_loudness],
                                             [u_f0_fit, u_loudness_fit])

        e_f0, e_loudness = inv_transform([e_f0, e_loudness],
                                         [e_f0_fit, e_loudness_fit])

        target = torch.cat([e_f0[:, 1:], e_loudness[:, 1:]], -1)
        model_out = torch.cat([out_f0, out_loudness], -1)

        e = Evaluator()
        score = e.evaluate(model_out, target, PLOT=True)
        out_audio = e.listen(model_out, target, ddsp,
                             wav_path + model_name + "-{}.wav".format(i))

        print("Score : ", score)

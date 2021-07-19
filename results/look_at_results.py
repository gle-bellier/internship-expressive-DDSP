import io as io
import scipy.io.wavfile as wav
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import soundfile as sf

#from get_datasets import get_datasets

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)

path = "results/diffusion/data/results.pickle"
number_of_examples = 3
# get data

with open(path, "rb") as dataset:
    dataset = pickle.load(dataset)

ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

# Initialize data :

n_sample = 2048

for i in range(number_of_examples):
    i *= n_sample
    u_f0 = dataset["u_f0"][i:i + n_sample]
    e_f0 = dataset["e_f0"][i:i + n_sample]
    pred_f0 = dataset["pred_f0"][i:i + n_sample]

    u_lo = dataset["u_lo"][i:i + n_sample]
    e_lo = dataset["e_lo"][i:i + n_sample]
    pred_lo = dataset["pred_lo"][i:i + n_sample]

    plt.plot(u_f0)
    plt.plot(e_f0)
    plt.plot(pred_f0)

    plt.show()

    plt.plot(u_lo)
    plt.plot(e_lo)
    plt.plot(pred_lo)

    plt.show()
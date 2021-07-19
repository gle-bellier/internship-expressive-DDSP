import io as io
import scipy.io.wavfile as wav
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

#from get_datasets import get_datasets

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)

import argparse
import os

path = os.path.abspath('../dataset/contours.csv')

parser = argparse.ArgumentParser(
    description="Create wav file from results data")
parser.add_argument("Path",
                    metavar="PATH",
                    type=str,
                    help="path to the results data")
parser.add_argument("Number",
                    metavar="Number of examples whished",
                    type=int,
                    help="Number of examples to compute")

args = parser.parse_args()

model_path = args.Path
number_of_examples = args.Number
print(model_path)
print(number_of_examples)

if not os.path.isfile(model_path):
    print('The file specified does not exist')

# get data

with open(path, "rb") as dataset:
    dataset = pickle.load(dataset)

ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

# Initialize data :

n_sample = 2048

for i in range(number_of_examples):
    u_f0 = dataset["u_f0"][i:i + n_sample]
    e_f0 = dataset["e_f0"][i:i + n_sample]
    pred_f0 = dataset["pred_f0"][i:i + n_sample]

    u_lo = dataset["u_lo"][i:i + n_sample]
    e_lo = dataset["e_lo"][i:i + n_sample]
    pred_lo = dataset["pred_lo"][i:i + n_sample]

    onsets = dataset["onsets"][i:i + n_sample]
    offsets = dataset["offsets"][i:i + n_sample]

    u_f0 = torch.from_numpy(u_f0).reshape(1, -1, 1).float()
    u_lo = torch.from_numpy(u_lo).reshape(1, -1, 1).float()

    e_f0 = torch.from_numpy(e_f0).reshape(1, -1, 1).float()
    e_lo = torch.from_numpy(e_lo).reshape(1, -1, 1).float()

    pred_f0 = torch.from_numpy(pred_f0).reshape(1, -1, 1).float()
    pred_lo = torch.from_numpy(pred_lo).reshape(1, -1, 1).float()

    onsets = torch.from_numpy(onsets).reshape(1, -1, 1).float()
    offsets = torch.from_numpy(offsets).reshape(1, -1, 1).float()

    midi = ddsp(u_f0, u_lo).reshape(-1).numpy()
    target = ddsp(e_f0, e_lo).reshape(-1).numpy()
    pred = ddsp(pred_f0, pred_lo).reshape(-1).numpy()

    sf.write("results/diffusion/samples/sample{}-pred.wav".format(i), pred,
             16000)
    sf.write("results/diffusion/samples/sample{}-midi.wav".format(i), midi,
             16000)
    sf.write("results/diffusion/samples/sample{}-resynth.wav".format(i),
             target, 16000)

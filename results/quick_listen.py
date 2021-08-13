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

instrument = "violin"
path = "results/baseline/data/results-{}-test.pickle".format(instrument)
number_of_examples = 5
# get data

with open(path, "rb") as dataset:
    dataset = pickle.load(dataset)

ddsp = torch.jit.load("ddsp_{}_pretrained.ts".format(instrument)).eval()

# Initialize data :

n_sample = 500

for i in range(number_of_examples):
    idx = i * n_sample
    u_f0 = dataset["u_f0"][idx:idx + n_sample]
    e_f0 = dataset["e_f0"][idx:idx + n_sample]
    pred_f0 = dataset["pred_f0"][idx:idx + n_sample]

    u_lo = dataset["u_lo"][idx:idx + n_sample]
    e_lo = dataset["e_lo"][idx:idx + n_sample]
    pred_lo = dataset["pred_lo"][idx:idx + n_sample]

    onsets = dataset["onsets"][idx:idx + n_sample]
    offsets = dataset["offsets"][idx:idx + n_sample]

    u_f0 = torch.from_numpy(u_f0).reshape(1, -1, 1).float()
    u_lo = torch.from_numpy(u_lo).reshape(1, -1, 1).float()

    e_f0 = torch.from_numpy(e_f0).reshape(1, -1, 1).float()
    e_lo = torch.from_numpy(e_lo).reshape(1, -1, 1).float()

    pred_f0 = torch.from_numpy(pred_f0).reshape(1, -1, 1).float()
    pred_lo = torch.from_numpy(pred_lo).reshape(1, -1, 1).float()

    onsets = torch.from_numpy(onsets).reshape(1, -1, 1).float()
    offsets = torch.from_numpy(offsets).reshape(1, -1, 1).float()

    midi = ddsp(u_f0, u_lo).reshape(-1).detach().numpy()
    target = ddsp(e_f0, e_lo).reshape(-1).detach().numpy()
    pred = ddsp(pred_f0, pred_lo).reshape(-1).detach().numpy()

    l = path.split("/")
    save_path = "/".join(l[:2])
    name = l[-1].split(".")[0] + str(i)

    sf.write("{}/samples/baseline-{}-pred.wav".format(save_path, name), pred,
             16000)
    #sf.write("{}/samples/{}-midi.wav".format(save_path, name), midi, 16000)
    #sf.write("{}/samples/{}-resynth.wav".format(save_path, name), target,
    #         16000)

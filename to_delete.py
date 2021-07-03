import torch
import pickle
from utils import *
from evaluation import Evaluator
import matplotlib.pyplot as plt

path = "dataset/dataset-article.pickle"

with open(path, "rb") as dataset:
    dataset = pickle.load(dataset)
n_sample = 500
idx = 64

u_f0 = dataset["u_f0"][idx:idx + n_sample]
u_loudness = dataset["u_loudness"][idx:idx + n_sample]
e_f0 = dataset["e_f0"][idx:idx + n_sample]
e_cents = dataset["e_cents"][idx:idx + n_sample]
e_loudness = dataset["e_loudness"][idx:idx + n_sample]

onsets = dataset["onsets"][idx:idx + n_sample]
offsets = dataset["offsets"][idx:idx + n_sample]

u_f0 = torch.from_numpy(u_f0).long().reshape(1, -1, 1)
u_loudness = torch.from_numpy(u_loudness).float().reshape(1, -1, 1)
e_f0 = torch.from_numpy(e_f0).long().reshape(1, -1, 1)
e_cents = torch.from_numpy(e_cents).float().reshape(1, -1, 1)
e_loudness = torch.from_numpy(e_loudness).float().reshape(1, -1, 1)

onsets = torch.from_numpy(onsets).float().reshape(1, -1, 1)
offsets = torch.from_numpy(offsets).float().reshape(1, -1, 1)

e = Evaluator()

trans, frames = e.get_trans_frames(onsets, offsets)

trans = e_cents * trans
frames = e_cents * frames

plt.plot(trans.squeeze())
plt.plot(frames.squeeze())
plt.show()
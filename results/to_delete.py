import torch
import pickle
from evaluation import Evaluator
import matplotlib.pyplot as plt

path = "dataset/dataset-article.pickle"

with open(path, "rb") as dataset:
    dataset = pickle.load(dataset)
n_sample = 500
idx = 64

u_f0 = dataset["u_f0"][idx:idx + n_sample]
u_lo = dataset["u_lo"][idx:idx + n_sample]
e_f0 = dataset["e_f0"][idx:idx + n_sample]
e_lo = dataset["e_lo"][idx:idx + n_sample]

pred_f0 = dataset["pred_f0"][idx:idx + n_sample]
pred_lo = dataset["pred_lo"][idx:idx + n_sample]

onsets = dataset["onsets"][idx:idx + n_sample]
offsets = dataset["offsets"][idx:idx + n_sample]

u_f0 = torch.from_numpy(u_f0).long().reshape(1, -1, 1)
u_lo = torch.from_numpy(u_lo).float().reshape(1, -1, 1)
e_f0 = torch.from_numpy(e_f0).long().reshape(1, -1, 1)
e_lo = torch.from_numpy(e_lo).float().reshape(1, -1, 1)

onsets = torch.from_numpy(onsets).float().reshape(1, -1, 1)
offsets = torch.from_numpy(offsets).float().reshape(1, -1, 1)

e = Evaluator()

trans, frames = e.get_trans_frames(onsets, offsets)

u_trans = u_f0 * trans
u_frames = u_f0 * frames

e_trans = e_f0 * trans
e_frames = e_f0 * frames

score_trans, score_frames = e.score(u_f0, u_lo, e_f0, e_lo, trans, frames)
score_total = score_trans + score_frames
print("Score frames = {}, score transitions = {}, score total = {}".format(
    score_frames, score_trans, score_total))

accuracy = e.accuracy(u_f0, u_f0 + 2, frames)
print("Accuracy = ", accuracy)

plt.plot(trans.squeeze())
plt.plot(frames.squeeze())
plt.show()
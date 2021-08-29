import torch
import pickle
from evaluation import Evaluator
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

path = "results/diffusion/data/violin-results.pickle"

with open(path, "rb") as results:
    results = pickle.load(results)
n_sample = 500
idx = 64

u_f0 = results["u_f0"][idx:idx + n_sample]
u_lo = results["u_lo"][idx:idx + n_sample]
e_f0 = results["e_f0"][idx:idx + n_sample]
e_lo = results["e_lo"][idx:idx + n_sample]

pred_f0 = results["pred_f0"][idx:idx + n_sample]
pred_lo = results["pred_lo"][idx:idx + n_sample]

onsets = results["onsets"][idx:idx + n_sample]
offsets = results["offsets"][idx:idx + n_sample]

u_f0 = torch.from_numpy(u_f0).float().reshape(1, -1, 1)
u_lo = torch.from_numpy(u_lo).float().reshape(1, -1, 1)
e_f0 = torch.from_numpy(e_f0).float().reshape(1, -1, 1)
e_lo = torch.from_numpy(e_lo).float().reshape(1, -1, 1)

pred_f0 = torch.from_numpy(pred_f0).float().reshape(1, -1, 1)
pred_lo = torch.from_numpy(pred_lo).float().reshape(1, -1, 1)

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

accuracy = e.accuracy(u_f0, e_f0, frames)
print("Accuracy = ", accuracy)

print(pred_f0.shape)
e.plot(pred_f0, pred_lo, e_f0, e_lo)

ddsp = torch.jit.load("ddsp_flute_pretrained.ts").eval()

model_audio, target_audio = e.listen(pred_f0,
                                     pred_lo,
                                     e_f0,
                                     e_lo,
                                     ddsp=ddsp,
                                     resynth=True)
e.plot_diff_spectrogram(model_audio, target_audio)
print(e.multi_scale_loss(model_audio, target_audio))
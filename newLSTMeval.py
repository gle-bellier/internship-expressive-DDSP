import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from torch.utils import data

torch.set_grad_enabled(False)
from expressive_dataset import ExpressiveDataset
from models.LSTMCategorical import FullModel
from newLSTMpreprocess import pctof
from effortless_config import Config
from random import randint
import soundfile as sf
from utils import Identity
from evaluation import Evaluator

import matplotlib.pyplot as plt


class args(Config):
    CKPT = None
    INFER_PITCH = False


args.parse_args()

ddsp = torch.jit.load("results/ddsp_debug_pretrained.ts").eval()

list_transforms = [
    (Identity, ),  # u_f0 
    (MinMaxScaler, ),  # u_loudness
    (Identity, ),  # e_f0
    (Identity, ),  # e_cents
    (MinMaxScaler, ),  # e_loudness
]

dataset = ExpressiveDataset(list_transforms=list_transforms)
model = FullModel.load_from_checkpoint(str(args.CKPT)).eval()

model_input, target = dataset[randint(0, len(dataset))]
model_input = model_input.unsqueeze(0).float()

f0, cents, loudness = model.generation_loop(model_input, args.INFER_PITCH)
target_f0, target_cents, target_loudness = target.split(1, -1)

d_cents = cents / 100 - .5
f0 = pctof(f0, d_cents)
loudness = loudness / (dataset.n_loudness - 1)

target_cents = target_cents / 100 - .5
target_f0 = pctof(target_f0, target_cents)
target_loudness = target_loudness / (dataset.n_loudness - 1)

f0 = dataset.apply_inverse_transform(f0.squeeze(0), 0)
loudness = dataset.apply_inverse_transform(loudness.squeeze(0), 1)

target_f0 = dataset.apply_inverse_transform(target_f0[1:], 2)
target_loudness = dataset.apply_inverse_transform(target_loudness[1:], 4)

y = ddsp(f0, loudness)
target_y = ddsp(target_f0, target_loudness)

name = str(args.CKPT).split("/")[1]
path = "results/saved_samples/"
sf.write(path + name + ".wav", y.reshape(-1).numpy(), 16000)
# name = "Essai"
# e = Evaluator()
# score = e.evaluate(f0,
#                    loudness,
#                    target_f0,
#                    target_loudness,
#                    reduction="median")
# print(score)
# out, target = e.listen(f0,
#                        loudness,
#                        target_f0,
#                        target_loudness,
#                        ddsp,
#                        "results/saved_samples/{}.wav".format(name),
#                        resynth=True)

# e.plot_diff_spectrogram(out, target)

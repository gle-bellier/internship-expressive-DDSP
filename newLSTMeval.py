import torch

torch.set_grad_enabled(False)
from newLSTMCat import FullModel, ExpressiveDataset
from csv_to_numpy import pctof
from effortless_config import Config
from random import randint
import soundfile as sf

import matplotlib.pyplot as plt


class args(Config):
    CKPT = None

args.parse_args()


ddsp = torch.jit.load("results/ddsp_debug_pretrained.ts").eval()

model = FullModel.load_from_checkpoint(args.CKPT).eval()
dataset = ExpressiveDataset()

model_input, target = dataset[randint(0, len(dataset))]
model_input = model_input.unsqueeze(0).float()

f0, cents, loudness = model.generation_loop(model_input)
cents = cents / 100 - .5

f0 = pctof(f0, cents)

loudness = loudness / dataset.n_loudness
loudness = dataset.unnormalize_loudness(loudness)

y = ddsp(f0.unsqueeze(-1), loudness.unsqueeze(-1))

sf.write("eval.wav", y.reshape(-1).numpy(), 16000)
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

torch.set_grad_enabled(False)
from newLSTMCat import FullModel, ExpressiveDataset
from newLSTMpreprocess import pctof
from effortless_config import Config
from random import randint
import soundfile as sf
from utils import Identity

import matplotlib.pyplot as plt


class args(Config):
    CKPT = None
    INFER_PITCH = False


args.parse_args()

ddsp = torch.jit.load("results/ddsp_debug_pretrained.ts").eval()

list_transforms = [
    (MinMaxScaler, ),  # u_f0 
    (QuantileTransformer, 31),  # u_loudness
    (MinMaxScaler, ),  # e_f0
    (Identity, ),  # e_cents
    (QuantileTransformer, 31),  # e_loudness
]

dataset = ExpressiveDataset(list_transforms=list_transforms)
model = FullModel.load_from_checkpoint(str(args.CKPT)).eval()

model_input, target = dataset[randint(0, len(dataset))]
model_input = model_input.unsqueeze(0).float()

f0, cents, loudness = model.generation_loop(model_input, args.INFER_PITCH)
cents = cents / 100 - .5

f0 = pctof(f0, cents)

loudness = loudness / (dataset.n_loudness - 1)

f0 = dataset.apply_inverse_transform(f0.squeeze(0), 1)
loudness = dataset.apply_inverse_transform(loudness.squeeze(0), 1)

y = ddsp(f0, loudness)

# sf.write("eval.wav", y.reshape(-1).numpy(), 16000)
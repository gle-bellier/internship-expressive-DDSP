import torch
import torch.nn as nn

torch.set_grad_enabled(False)
from redif.model import Model, ftom
from effortless_config import Config
from os import path, makedirs
from glob import glob
import soundfile as sf

import numpy as np


class args(Config):
    CKPT = None
    DDSP = None
    OUT = "redif_samples/"


args.parse_args()

makedirs(args.OUT, exist_ok=True)
n_sample = len(glob(path.join(args.OUT, "*.wav")))
sample_len = 500
nb_samples = 10

# INSTANCIATE MODELS
model = Model.load_from_checkpoint(args.CKPT, strict=False).eval()
model.set_noise_schedule()
ddsp = torch.jit.load(args.DDSP).eval()

################################################
# TODO: replace dummy input with actual contours

patch = "redifcontours/"
name = "flute-test"

pitch_all = torch.from_numpy(np.load("{}u_f0{}.npy".format(path,
                                                           name))).reshape(
                                                               1, -1).long()
loudness_all = torch.from_numpy(np.load("{}u_lo{}.npy".format(
    path, name))).reshape(1, -1).float()
################################################

for i in range(nb_samples):

    pitch = pitch_all[sample_len * i:sample_len * (i + 1)]
    loudness = loudness_all[sample_len * i:sample_len * (i + 1)]
    # FORMAT INPUT CONTOURS
    pitch = nn.functional.one_hot(pitch, 127).permute(0, 2, 1)
    loudness = loudness.unsqueeze(1)

    assert torch.all(pitch >= 0) and torch.all(pitch < 128), str(pitch)

    env = torch.cat([pitch, loudness], 1)
    x = torch.randn(env.shape[0], model.data_dim, env.shape[-1])

    # SAMPLE FROM ESTIMATED DISTRIBUTION
    y = model.sample(x, env)
    f0, lo = model.transform.inverse(y)

    # SYNTHESIS
    sound = ddsp(f0.permute(0, 2, 1), lo.permute(0, 2, 1))
    name = f"{name}_{i + n_sample:03d}.wav"
    sf.write(path.join(args.OUT, name), sound.reshape(-1).numpy(), 16000)
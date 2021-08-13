import numpy as np
from effortless_config import Config
import pickle
from random import shuffle
from os import path


class args(Config):
    DATA = None
    LENGHT = 512
    N_SAMPLES = 10


args.parse_args()

with open(args.DATA, "rb") as data:
    data = pickle.load(data)

u_f0 = np.asarray(data["u_f0"]).reshape(-1)
u_lo = np.asarray(data["u_loudness"]).reshape(-1)
e_f0 = np.asarray(data["e_f0"]).reshape(-1)
e_lo = np.asarray(data["e_loudness"]).reshape(-1)
onsets = np.asarray(data["onsets"]).reshape(-1)
offsets = np.asarray(data["offsets"]).reshape(-1)

len_data = len(u_f0) // args.LENGHT
idx = list(range(len_data))
shuffle(idx)

idx = idx[:args.N_SAMPLES]

u_f0 = np.stack([u_f0[i * args.LENGHT:(i + 1) * args.LENGHT] for i in idx])
u_lo = np.stack([u_lo[i * args.LENGHT:(i + 1) * args.LENGHT] for i in idx])
e_f0 = np.stack([e_f0[i * args.LENGHT:(i + 1) * args.LENGHT] for i in idx])
e_lo = np.stack([e_lo[i * args.LENGHT:(i + 1) * args.LENGHT] for i in idx])
onsets = np.stack([onsets[i * args.LENGHT:(i + 1) * args.LENGHT] for i in idx])
offsets = np.stack(
    [offsets[i * args.LENGHT:(i + 1) * args.LENGHT] for i in idx])

data = {
    "u_f0": u_f0.reshape(-1),
    "u_loudness": u_lo.reshape(-1),
    "e_f0": e_f0.reshape(-1),
    "e_loudness": e_lo.reshape(-1),
    "onsets": onsets.reshape(-1),
    "offsets": offsets.reshape(-1),
}

name = path.splitext(path.basename(args.DATA))[0]
out_name = f"{name}_shuffled_cropped.pickle"

with open(path.join("dataset", out_name), "wb") as out:
    pickle.dump(data, out)
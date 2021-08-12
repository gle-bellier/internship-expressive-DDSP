import numpy as np
import matplotlib.pyplot as plt
from effortless_config import Config
import pickle as pk
from os import path, makedirs


class args(Config):
    DATA = "dataset/test-set.pickle"


args.parse_args()


def ftom(f):
    return 12 * (np.log(f) - np.log(440)) / np.log(2) + 69


ext = path.splitext(args.DATA)[-1]
name = path.splitext(path.basename(args.DATA))[0]

if ext == ".csv":
    with open(args.DATA, "r") as contours:
        contours = contours.read()

    contours = contours.split("\n")
    print(f"extracting {contours.pop(0).split(',')[:4]}")
    contours.pop(-1)

    data = {
        "u_f0": [],
        "u_lo": [],
        "e_f0": [],
        "e_lo": [],
    }

    for t in contours:
        u_f0, u_loudness, e_f0, e_loudness = t.split(",")[:4]
        data["u_f0"].append(float(u_f0))
        data["u_lo"].append(float(u_loudness))
        data["e_f0"].append(float(e_f0))
        data["e_lo"].append(float(e_loudness))

    u_f0 = np.asarray(data["u_f0"])
    u_lo = np.asarray(data["u_lo"])
    e_f0 = np.asarray(data["e_f0"])
    e_lo = np.asarray(data["e_lo"])

elif ext == ".pickle":
    with open(args.DATA, "rb") as contours:
        contours = pk.load(contours)
    u_f0 = np.asarray(contours["u_f0"]).reshape(-1)
    u_lo = np.asarray(contours["u_loudness"]).reshape(-1)
    e_f0 = np.asarray(contours["e_f0"]).reshape(-1)
    e_lo = np.asarray(contours["e_loudness"]).reshape(-1)

else:
    raise Exception(f"data type {ext} not understood")

u_f0 = np.round(ftom(u_f0)).astype(int)
u_lo = np.clip(u_lo, -6, None)
e_f0 = np.clip(e_f0, 1e-5, None)

print(f"loaded 4 arrays from {ext} file")
print(f"u_f0: {u_f0.shape}, mean: {np.mean(u_f0)}, std: {np.std(u_f0)}")
print(f"u_lo: {u_lo.shape}, mean: {np.mean(u_lo)}, std: {np.std(u_lo)}")
print(f"e_f0: {e_f0.shape}, mean: {np.mean(e_f0)}, std: {np.std(e_f0)}")
print(f"e_lo: {e_lo.shape}, mean: {np.mean(e_lo)}, std: {np.std(e_lo)}")

assert all(u_f0 >= 0) and all(u_f0 < 128)

out_dir = path.join("redif_contours", name)
makedirs(out_dir, exist_ok=True)

np.save(path.join(out_dir, "u_f0.npy"), u_f0)
np.save(path.join(out_dir, "u_lo.npy"), u_lo)
np.save(path.join(out_dir, "e_f0.npy"), e_f0)
np.save(path.join(out_dir, "e_lo.npy"), e_lo)

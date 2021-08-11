import numpy as np
import matplotlib.pyplot as plt
from effortless_config import Config


class args(Config):
    CSV = "dataset/contours.csv"


args.parse_args()


def ftom(f):
    return 12 * (np.log(f) - np.log(440)) / np.log(2) + 69


with open(args.CSV, "r") as contours:
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

u_f0 = np.round(ftom(u_f0)).astype(int)

assert all(u_f0 >= 0) and all(u_f0 < 128)
assert all(e_f0 > 0)

np.save("u_f0.npy", u_f0)
np.save("u_lo.npy", u_lo)
np.save("e_f0.npy", e_f0)
np.save("e_lo.npy", e_lo)

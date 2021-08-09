import numpy as np
import matplotlib.pyplot as plt


def ftom(f):
    return np.round(12 * (np.log(f) - np.log(440)) / np.log(2) + 69)


with open("dataset/contours.csv", "r") as contours:
    contours = contours.read()

contours = contours.split("\n")
contours.pop(0)
contours.pop(-1)

data = {
    "u_f0": [],
    "u_lo": [],
    "e_f0": [],
    "e_lo": [],
}

for t in contours:
    u_f0, u_loudness, e_f0, e_loudness, _, _, _, _ = t.split(",")
    data["u_f0"].append(float(u_f0))
    data["u_lo"].append(float(u_loudness))
    data["e_f0"].append(float(e_f0))
    data["e_lo"].append(float(e_loudness))

data["u_f0"] = np.asarray(data["u_f0"])
data["u_lo"] = np.asarray(data["u_lo"])
data["e_f0"] = np.asarray(data["e_f0"])
data["e_lo"] = np.asarray(data["e_lo"])

np.save("u_f0.npy", data["u_f0"])
np.save("u_lo.npy", data["u_lo"])
np.save("e_f0.npy", data["e_f0"])
np.save("e_lo.npy", data["e_lo"])

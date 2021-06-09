import csv
import numpy as np
import pickle

u_f0 = []
u_loudness = []
e_f0 = []
e_loudness = []

with open("dataset/contours.csv", "r") as contour:
    contour = csv.DictReader(contour)

    for row in contour:
        u_f0.append(row["u_f0"])
        u_loudness.append(row["u_loudness"])
        e_f0.append(row["e_f0"])
        e_loudness.append(row["e_loudness"])

u_f0 = np.asarray(u_f0).astype(float)
u_loudness = np.asarray(u_loudness).astype(float)
e_f0 = np.asarray(e_f0).astype(float)
e_loudness = np.asarray(e_loudness).astype(float)


def mtof(m):
    return 440 * 2**((m - 69) / 12)


def ftom(f):
    return 12 * (np.log(f) - np.log(440)) / np.log(2) + 69


def norm_array(x):
    minimum = np.min(x)
    maximum = np.max(x)
    x = (x - minimum) / (maximum - minimum)
    return x, minimum, maximum


def ftopc(f):
    m_float = ftom(f)
    m_int = np.round(m_float).astype(int)
    c_float = m_float - m_int
    return m_int, c_float


e_f0, e_cents = ftopc(e_f0)
u_f0, _ = ftopc(u_f0)

print(np.max(e_f0))
print(np.max(u_f0))

out = {
    "u_f0": u_f0, # 0 - 127
    "u_loudness": norm_array(u_loudness), # 0 - 1
    "e_f0": e_f0, # 0 - 127
    "e_cents": e_cents + .5, # 0 - 1
    "e_loudness": norm_array(e_loudness), # 0 - 1
}

with open("dataset.pickle", "wb") as file_out:
    pickle.dump(out, file_out)


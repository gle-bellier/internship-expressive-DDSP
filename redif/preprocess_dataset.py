import numpy as np
import matplotlib.pyplot as plt
import pickle


def ftom(f):
    return 12 * (np.log(f) - np.log(440)) / np.log(2) + 69


instrument = "violin"
path = "dataset/{}-train-da.pickle".format(instrument)
print("dataset file used : {}".format(path))
print("Loading Dataset...")
with open(path, "rb") as dataset:
    dataset = pickle.load(dataset)

u_f0 = np.asarray(dataset["u_f0"])
u_lo = np.asarray(dataset["u_loudness"])
e_f0 = np.asarray(dataset["e_f0"])
e_lo = np.asarray(dataset["e_loudness"])

u_f0 = np.round(ftom(u_f0)).astype(int).clip(0, 127)
e_f0 = np.clip(e_f0, 1e-4, None)

plt.plot(u_f0[:5000])
plt.plot(e_f0[:5000])
plt.show()

assert all(u_f0 >= 0) and all(u_f0 < 128)
assert all(e_f0 > 0)

np.save("u_f0.npy", u_f0)
np.save("u_lo.npy", u_lo)
np.save("e_f0.npy", e_f0)
np.save("e_lo.npy", e_lo)

import csv
import matplotlib.pyplot as plt
import numpy as np


PATH = "dataset/contours.csv"

u_f0 = []
u_loudness = []
e_f0 = []
e_loudness = []
e_f0_mean = []
e_f0_stddev = []

with open(PATH) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        u_f0.append(float(row["u_f0"]))
        u_loudness.append(float(row["u_loudness"]))
        e_f0.append(float(row["e_f0"]))
        e_loudness.append(float(row["e_loudness"]))
        e_f0_mean.append(float(row["e_f0_mean"]))
        e_f0_stddev.append(float(row["e_f0_stddev"]))


u_f0 = np.array(u_f0)
u_loudness = np.array(u_loudness)
e_f0 = np.array(e_f0)
e_loudness = np.array(e_loudness)
e_f0_mean = np.array(e_f0_mean)
e_f0_stddev = np.array(e_f0_stddev)


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(u_f0.squeeze()[8000:10000], label = "u_f0")
ax1.plot(e_f0.squeeze()[8000:10000], label = "e_f0")
ax1.legend()

ax2.plot(u_loudness.squeeze()[8000:10000], label = "u_loudness")
ax2.plot(e_loudness.squeeze()[8000:10000], label = "e_loudness")
ax2.legend()

plt.legend()
plt.show()
import csv
import matplotlib.pyplot as plt
import numpy as np

PATH = "dataset/contours-article.csv"

u_f0 = []
u_loudness = []
e_f0 = []
e_loudness = []
e_f0_mean = []
e_f0_stddev = []
f0_conf = []
events = []

with open(PATH) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        u_f0.append(float(row["u_f0"]))
        u_loudness.append(float(row["u_loudness"]))
        e_f0.append(float(row["e_f0"]))
        e_loudness.append(float(row["e_loudness"]))
        e_f0_mean.append(float(row["e_f0_mean"]))
        e_f0_stddev.append(float(row["e_f0_stddev"]))
        f0_conf.append(float(row["f0_conf"]))
        events.append(float(row["events"]))

u_f0 = np.array(u_f0)
u_loudness = np.array(u_loudness)
e_f0 = np.array(e_f0)
e_loudness = np.array(e_loudness)
e_f0_mean = np.array(e_f0_mean)
e_f0_stddev = np.array(e_f0_stddev)
f0_conf = np.array(f0_conf)
events = np.array(events)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
ax1.plot(u_f0.squeeze()[8000:10000], label="u_f0")
ax1.plot(e_f0.squeeze()[8000:10000], label="e_f0")
ax1.legend()

ax2.plot(u_loudness.squeeze()[8000:14000], label="u_loudness")
ax2.plot(e_loudness.squeeze()[8000:14000], label="e_loudness")
ax2.legend()

ax3.plot(
    (e_f0_mean.squeeze()[8000:10000] / np.max(e_f0_mean[8000:10000])) * 50,
    label="e_mean ")
ax3.plot(e_f0_stddev.squeeze()[8000:10000].clip(-50, 50), label="e_std")
ax3.legend()

ax4.plot(u_f0.squeeze()[8000:10000] / np.max(u_f0[8000:10000]), label="u_f0")
ax4.plot(events.squeeze()[8000:10000], label="events")
ax4.legend()

ax5.plot(u_f0.squeeze()[8000:10000] / np.max(u_f0[8000:10000]), label="u_f0")
ax5.plot(f0_conf.squeeze()[8000:10000], label="f0_conf")
ax5.legend()

plt.legend()
plt.show()
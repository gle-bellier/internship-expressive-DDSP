import csv
import matplotlib.pyplot as plt
import numpy as np

PATHS = ["dataset/contours.csv", "dataset/contours-violin-update.csv"]

u_f0 = []
u_loudness = []
e_f0 = []
e_loudness = []
e_f0_mean = []
e_f0_stddev = []
f0_conf = []
events = []

for PATH in PATHS:
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

print("Dataset length : {}min {}s".format((len(u_f0) // 100) // 60,
                                          (len(u_f0) // 100) % 60))

with open('dataset/all-violin-contours-updated.csv', 'w',
          newline='') as csvfile:
    fieldnames = [
        "u_f0", "u_loudness", "e_f0", "e_loudness", "e_f0_mean", "e_f0_stddev",
        "f0_conf", "events"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for i in range(len(u_f0)):
        writer.writerow({
            "u_f0": u_f0[i],
            "u_loudness": u_loudness[i],
            "e_f0": e_f0[i],
            "e_loudness": e_loudness[i],
            "e_f0_mean": e_f0_mean[i],
            "e_f0_stddev": u_f0[i],
            "f0_conf": f0_conf[i],
            "events": events[i]
        })

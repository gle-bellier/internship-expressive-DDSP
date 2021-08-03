import io as io
import scipy.io.wavfile as wav
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pickle
import soundfile as sf

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)

path = "results/diffusion/data/results.pickle"
number_of_examples = 5
# get data

dataset = pd.read_pickle(path, compression=None)

df = pd.DataFrame(dataset)
print(df)

# sns.set_theme(style="darkgrid")

sns.lineplot(x="time", y="u_f0", hue="sample", data=df[df["sample"] == 0])

sns.lineplot(x="time", y="e_f0", hue="sample", data=df[df["sample"] == 0])
plt.show()

# ddsp = torch.jit.load("ddsp_violin_pretrained.ts").eval()

# # Initialize data :

# n_sample = 2048

# for i in range(number_of_examples):
#     i *= n_sample
#     u_f0 = dataset["u_f0"][i:i + n_sample]
#     e_f0 = dataset["e_f0"][i:i + n_sample]
#     pred_f0 = dataset["pred_f0"][i:i + n_sample]

#     u_lo = dataset["u_lo"][i:i + n_sample]
#     e_lo = dataset["e_lo"][i:i + n_sample]
#     pred_lo = dataset["pred_lo"][i:i + n_sample]

#     plt.plot(u_f0, label="midi")
#     plt.plot(e_f0, label="target")
#     plt.plot(pred_f0, label="model")
#     plt.legend()
#     plt.show()

#     plt.plot(u_lo, label="Midi")
#     plt.plot(e_lo, label="Target")
#     plt.plot(pred_lo, label="Model")
#     plt.legend()
#     plt.show()
import io as io
import scipy.io.wavfile as wav
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import soundfile as sf

#from get_datasets import get_datasets

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)

path = "results/diffusion/data/results.pickle"
# get data

with open(path, "rb") as dataset:
    dataset = pickle.load(dataset)

ddsp = torch.jit.load("ddsp_violin_pretrained.ts").eval()

# Initialize data :

n_sample = 2048

u_f0 = dataset["u_f0"]
e_f0 = dataset["e_f0"]
pred_f0 = dataset["pred_f0"]

u_lo = dataset["u_lo"]
e_lo = dataset["e_lo"]
pred_lo = dataset["pred_lo"]

shift = pred_f0 - u_f0

sns.displot(shift)
plt.show()
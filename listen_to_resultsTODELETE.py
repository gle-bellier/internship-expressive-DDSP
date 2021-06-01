import io as io
import scipy.io.wavfile as wav

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler

from get_datasets import get_datasets


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)





save_path = "results/saved_models/"
model_name = "LSTM_towards_realistic_midi.pth"


ddsp = torch.jit.load("results/ddsp_debug_pretrained.ts")

import numpy as np 
import matplotlib.pyplot as plt

number_samples = 5


with open("results/results.npy", "rb") as f:
    for i in range(number_samples):
        try: 
            t = np.load(f)
            u_f0 = np.load(f)
            e_f0 = np.load(f)
            out_f0 = np.load(f)
            u_loudness = np.load(f)
            e_loudness = np.load(f)
            out_loudness = np.load(f)
            
        except:
            break
    

        fig, (ax1, ax2) = plt.subplots(2, 1)

    
        ax1.plot(t, out_f0, label = "Model")
        ax1.plot(t, u_f0, label = "Midi")
        ax1.plot(t, e_f0, label = "Performance")
        ax1.legend()
        
        ax2.plot(t, out_loudness, label = "Model")
        ax2.plot(t, u_loudness, label = "Midi")
        ax2.plot(t, e_loudness, label = "Performance")
        ax2.legend()

        plt.legend()
        plt.show()


        e_f0, e_loudness = torch.Tensor(e_f0).unsqueeze(0).unsqueeze(2), torch.Tensor(e_loudness).unsqueeze(0).unsqueeze(2)
        audio = ddsp(e_f0, e_loudness).detach().squeeze().numpy()

        print(audio.shape)

        sampling_rate = 16000
        filename = "{}-sample{}.wav".format(model_name, i)

        wav.write(filename, sampling_rate, audio.astype(np.int16))


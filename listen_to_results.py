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


PATH = save_path + model_name 
model = torch.load(PATH, map_location=device)
print(model.parameters)


sc = StandardScaler()
sampling_rate = 100
number_of_examples = 50



_, test_loader = get_datasets(dataset_file = "dataset/contours.csv", sampling_rate = sampling_rate, sample_duration = 20, batch_size = 1, ratio = 0.7, transform=sc.fit_transform)
test_data = iter(test_loader)




ddsp = torch.jit.load("results/ddsp_debug_pretrained.ts")


model.eval()
with torch.no_grad():
    for i in range(number_of_examples):
        print("Sample {} reconstruction".format(i))

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = next(test_data)

        u_f0 = torch.Tensor(u_f0.float()) 
        u_loudness = torch.Tensor(u_loudness.float())
        


        e_f0, e_loudness = model.predict(u_f0, u_loudness)


        audio = ddsp(e_f0, e_loudness).detach().squeeze().numpy()

        print(audio.shape)


        sampling_rate = 16000
        filename = "{}-sample{}.wav".format(model_name, i)

        wav.write(filename, sampling_rate, audio.astype(np.int16))


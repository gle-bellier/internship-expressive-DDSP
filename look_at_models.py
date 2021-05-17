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




save_path = "models/saved_models/"
model_name = "LSTM_towards_realistic_midi.pth"


PATH = save_path + model_name 
model = torch.load(PATH, map_location=device)
print(model.parameters)


sc = StandardScaler()
sampling_rate = 100

_, test_loader = get_datasets(dataset_file = "dataset/contours.csv", sampling_rate = sampling_rate, sample_duration = 20, batch_size = 1, ratio = 0.7, transform=sc.fit_transform)
test_data = iter(test_loader)

number_of_examples = 20

model.eval()
with torch.no_grad():

    for i in range(number_of_examples):

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = next(test_data)
        u_f0 = torch.Tensor(u_f0.float())
        u_loudness = torch.Tensor(u_loudness.float())
        e_f0 = torch.Tensor(e_f0.float())
        e_loudness = torch.Tensor(e_loudness.float())
        e_f0_mean = torch.Tensor(e_f0_mean.float())
        e_f0_stddev = torch.Tensor(e_f0_stddev.float())


        model_input = torch.cat([
                u_f0[:, 1:],
                u_loudness[:, 1:],
                e_f0[:, :-1],
                e_loudness[:, :-1]            
                ], -1)

        output = model(model_input.to(device))


        out_f0, out_loudness = torch.split(output, 1, -1)
        out_f0, out_loudness = out_f0.squeeze().detach(), out_loudness.squeeze().detach()

        perf_f0, perf_loudness = e_f0[:,1:].squeeze().detach(), e_loudness[:,1:].squeeze().detach()
        t = np.array([i/sampling_rate for i in range(out_f0.shape[0])])
        
        
        

        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, out_f0, label = "Model")
        ax1.plot(t, perf_f0, label = "Performance")
        ax1.legend()
        
        ax2.plot(t, out_loudness, label = "Model")
        ax2.plot(t, perf_loudness, label = "Performance")
        ax2.legend()

        plt.legend()
        plt.show()



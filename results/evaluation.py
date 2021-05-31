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

number_of_examples = 50



def get_model_next_step(model, u_f0, u_loudness, e_f0, e_loudness):

        model_input = torch.cat([u_f0, u_loudness, e_f0, e_loudness], -1)
        output = model(model_input.to(device))
        out_f0, out_loudness = torch.split(output, 1, -1)

        out_f0 = torch.cat([e_f0[:,1:], out_f0[:,-1:]], 1)
        out_loudness = torch.cat([e_loudness[:,1:], out_loudness[:,-1:]], 1)

        return out_f0, out_loudness
    


model.eval()
with torch.no_grad():
    with open("results.npy", "wb") as f:

        for i in range(number_of_examples):
            print("Sample {} reconstruction".format(i))

            init_u_f0, init_u_loudness, init_e_f0, init_e_loudness, init_e_f0_mean, init_e_f0_stddev = next(test_data)
            u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = next(test_data)




            u_f0 = torch.cat([torch.Tensor(init_u_f0.float()), torch.Tensor(u_f0.float())], 1) 
            u_loudness = torch.cat([torch.Tensor(init_u_loudness.float()), torch.Tensor(u_loudness.float())], 1)
            
            e_f0 = torch.Tensor(e_f0.float())
            e_loudness = torch.Tensor(e_loudness.float())



            out_f0 = e_f0[:, 1:]
            out_loudness = e_loudness[:, 1:]


            n_step = 2000
            for j in range(n_step):
                out_f0, out_loudness = get_model_next_step(model, u_f0[:, j:1999+j].to(device), u_loudness[:, j:1999+j].to(device), out_f0.to(device), out_loudness.to(device))


            out_f0, out_loudness = out_f0.squeeze().detach(), out_loudness.squeeze().detach()
            e_f0, e_loudness = e_f0[:,1:].squeeze().detach(), e_loudness[:,1:].squeeze().detach()
            u_f0, u_loudness = u_f0.squeeze().detach()[-1999:], u_loudness.squeeze().detach()[-1999:]

            t = np.array([i/sampling_rate for i in range(out_f0.shape[0])])
            
            np.save(f, t)
            np.save(f, u_f0.cpu())
            np.save(f, e_f0.cpu())
            np.save(f, out_f0.cpu())
            np.save(f, u_loudness.cpu())
            np.save(f, e_loudness.cpu())
            np.save(f, out_loudness.cpu())
            


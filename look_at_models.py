import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

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
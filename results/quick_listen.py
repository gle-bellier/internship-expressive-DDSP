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

#from get_datasets import get_datasets


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)

import argparse
import os


path =  os.path.abspath('../dataset/contours.csv')

parser = argparse.ArgumentParser(description="Create wav file from dataset according to a given model")
parser.add_argument("Path", metavar="PATH", type=str, help="path to the model")
parser.add_argument("Number", metavar="Number of examples whished", type=int, help="Number of examples to compute")

args = parser.parse_args()


model_path = args.Path
number_of_examples = args.Number
print(model_path)
print(number_of_examples)



if not os.path.isfile(model_path):
    print('The file specified does not exist')


save_path = "models/saved_models/"
model_name = "LSTM_towards_realistic_midi.pth"


PATH = save_path + model_name 
model = torch.load(PATH, map_location=device)
print(model.parameters)





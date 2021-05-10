
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt



class LSTMContours(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMContours, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


        self.conv = nn.Conv1d(self.input_size, self.input_size*2, kernel_size=3)
        self.lstm = nn.LSTMCell(self.input_size*2, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt



class LSTMContours(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(LSTMContours, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.conv1d = nn.Conv1d(in_channels = 2000, out_channels=2000, kernel_size=5, stride=1, dilation=4)



        self.lstm = nn.LSTM(input_size=input_size//2, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        x = self.conv1d(x)
        #print("Size x ", x.shape)
        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        #out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)

        return out
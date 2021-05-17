
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt



class LSTMContours(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMContours, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(4, 16)
        self.lin2 = nn.Linear(16, input_size)

        self.lkrelu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(1999) # 2000 -1 (delay for prediction)


        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
       
        x = self.lin1(x)
        x = self.lkrelu(x)
        x = self.lin2(x)
        x = self.lkrelu(x)

        x = self.bn(x)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)

        
        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        #out = out.contiguous().view(-1, self.hidden_size)
        out = self.lkrelu(out)


        out = self.fc1(out)
        x = self.lkrelu(x)
        x = self.bn(x)

        out = self.fc2(out)
        x = self.lkrelu(x)
        out = self.fc3(out)
        x = self.lkrelu(x)

        return out
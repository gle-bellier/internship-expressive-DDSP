
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np



class LSTMContours(nn.Module):
    
    def __init__(self, input_size = 256, hidden_size = 512, num_layers = 1):
        super(LSTMContours, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(4, 64)
        self.lin2 = nn.Linear(64, 256)

        self.lkrelu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(1999) # 2000 -1 (delay for prediction)


        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
       
        x = self.lin1(x)
        x = self.lkrelu(x)
        x = self.bn(x)

        x = self.lin2(x)
        x = self.lkrelu(x)
        x = self.bn(x)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)

        
        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        #out = out.contiguous().view(-1, self.hidden_size)

        
        out = self.fc1(out)
        out = self.lkrelu(out)
        out = self.bn(out)

        out = self.fc2(out)
        out = self.lkrelu(out)
        out = self.bn(out)
        
        out = self.fc3(out)

        return out
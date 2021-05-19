import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt



class LSTMContoursBCE(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMContoursBCE, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(1, 4)
        self.lin2 = nn.Linear(4, 8)
        self.lin3 = nn.Linear(8, input_size)
        self.lkrelu = nn.LeakyReLU()

        self.sig = nn.Sigmoid()


        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(hidden_size, 200)


    def forward(self, x):
       
        x = self.lin1(x)
        x = self.lkrelu(x)
        x = self.lin2(x)
        x = self.lkrelu(x)
        x = self.lin3(x)
        x = self.lkrelu(x)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)

        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        #out = out.contiguous().view(-1, self.hidden_size)
        out = self.lkrelu(out)
        out = self.fc1(out)
        out = self.lkrelu(out)
        out = self.fc2(out)

        cents, pitch = torch.split(out, 100, dim = -1)

        return pitch, cents



class LSTMContoursMSE(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMContoursMSE, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(1, 4)
        self.lin2 = nn.Linear(4, 8)
        self.lin3 = nn.Linear(8, input_size)
        self.lkrelu = nn.LeakyReLU()

        self.sig = nn.Sigmoid()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)


    def forward(self, x):
       
        x = self.lin1(x)
        x = self.lkrelu(x)
        x = self.lin2(x)
        x = self.lkrelu(x)
        x = self.lin3(x)
        x = self.lkrelu(x)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)

        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        #out = out.contiguous().view(-1, self.hidden_size)
        out = self.lkrelu(out)
        out = self.fc1(out)
        out = self.lkrelu(out)
        out = self.fc2(out)

        out = self.sig(out)

        return out
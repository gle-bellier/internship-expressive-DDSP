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
        
        self.lin = nn.Linear(1, input_size)
        self.lkrelu = nn.LeakyReLU()

        self.sig = nn.Sigmoid()
        self.sm = nn.Softmax(dim=-1)


        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 200)

    def forward(self, x):
       
        x = self.lin(x)
        x = self.lkrelu(x)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)

        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        #out = out.contiguous().view(-1, self.hidden_size)
        out = self.lkrelu(out)
        out = self.fc(out)

        out = self.sig(out)
        out = self.sm(out)


        cents, pitch = torch.split(out, 100, dim = -1)

        return pitch, cents
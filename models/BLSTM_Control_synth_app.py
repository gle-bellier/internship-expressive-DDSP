
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


        def dense_block(a, b):
            return nn.Sequential(
            nn.Linear(a, b),
            nn.BatchNorm1d(2000),
            nn.LeakyReLU(),
            )
        
        self.dense1a = dense_block(1, input_size//2)
        self.dense1b = dense_block(1, input_size//2)
        self.dense2 = dense_block(hidden_size + 2, 32)

        self.lin1 = nn.Linear(32, 2)
        self.lin2 = nn.Linear(1, 1)
        
        #self.bn = nn.BatchNorm
        self.sig = nn.Sigmoid()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional = False,
                            batch_first=True)


    def forward(self, x_pitch, x_velocity):
       
        x_p = self.dense1a(x_pitch)
        x_v = self.dense1b(x_velocity)

        x = torch.cat([x_p, x_v], -1)
        
        # Initialize BLSTM states 
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)        
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)

        # Propagate input through BLSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        out = torch.cat([x_pitch, out, x_velocity], -1)

        out = self.dense2(out)        
        out = self.lin1(out)
    
        y_pitch, y_velocity = torch.split(out, 1, dim = -1)

        pitch_contour = self.sig(y_pitch) + self.lin2(x_pitch)
        loudness_contour = self.sig(y_velocity) + self.lin2(x_velocity)

        out = torch.cat([pitch_contour, loudness_contour], -1)

        return out





class BLSTMContours(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(BLSTMContours, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size


        def dense_block(a, b):
            return nn.Sequential(
            nn.Linear(a, b),
            nn.BatchNorm1d(2000),
            nn.LeakyReLU(),
            )
        
        self.dense1a = dense_block(1, input_size//2)
        self.dense1b = dense_block(1, input_size//2)
        self.dense2 = dense_block(hidden_size * 2 + 2, 32)

        self.lin1 = nn.Linear(32, 2)
        self.lin2 = nn.Linear(1, 1)
        
        #self.bn = nn.BatchNorm
        self.sig = nn.Sigmoid()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional = True,
                            batch_first=True)


    def forward(self, x_pitch, x_velocity):
       
        x_p = self.dense1a(x_pitch)
        x_v = self.dense1b(x_velocity)

        x = torch.cat([x_p, x_v], -1)
        
        # Initialize BLSTM states 
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device = x.device)        
        c_0 = torch.zeros(self.num_layers *2, x.size(0), self.hidden_size, device = x.device)

        # Propagate input through BLSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        out = torch.cat([x_pitch, out, x_velocity], -1)

        out = self.dense2(out)        
        out = self.lin1(out)
    
        y_pitch, y_velocity = torch.split(out, 1, dim = -1)

        pitch_contour = self.sig(y_pitch) + self.lin2(x_pitch)
        loudness_contour = self.sig(y_velocity) + self.lin2(x_velocity)

        out = torch.cat([pitch_contour, loudness_contour], -1)

        return out
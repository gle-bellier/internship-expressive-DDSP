
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

        self.bn1 = nn.BatchNorm1d(64) 
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256) 
        self.bn4 = nn.BatchNorm1d(64)


        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def initialise_h0_c0(self, x):
        
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        return h_0, c_0



    def forward(self, x):
       
        x = self.lin1(x)
        x = self.lkrelu(x)

        x = x.transpose(1, 2)
        x = self.bn1(x)
        x = x.transpose(1, 2)

        x = self.lin2(x)
        x = self.lkrelu(x)

        x = x.transpose(1, 2)
        x = self.bn2(x)
        x = x.transpose(1, 2)

        h_0, c_0 = self.initialise_h0_c0(x)

        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))

        
        out = self.fc1(out)
        out = self.lkrelu(out)

        out = out.transpose(1, 2)
        out = self.bn3(out)
        out = out.transpose(1, 2)

        out = self.fc2(out)
        out = self.lkrelu(out)

        out = out.transpose(1, 2)
        out = self.bn4(out)
        out = out.transpose(1, 2)
        
        out = self.fc3(out)

        return out



    def predict(self, pitch, loudness):

        f0 = torch.zeros_like(pitch)
        l0 = torch.zeros_like(loudness)

        x_in = torch.cat([pitch, loudness, f0, l0], -1)
        
        h_t = torch.zeros(self.num_layers, 1, self.hidden_size, device = x_in.device)
        c_t = torch.zeros(self.num_layers, 1, self.hidden_size, device = x_in.device)

        for i in range(x_in.size(1)):

            x = self.lin1(x_in)
            x = self.lkrelu(x)

            x = x.transpose(1, 2)
            x = self.bn1(x)
            x = x.transpose(1, 2)

            x = self.lin2(x)
            x = self.lkrelu(x)

            x = x.transpose(1, 2)
            x = self.bn2(x)
            x = x.transpose(1, 2)
        
            pred, (h_t, c_t)  = self.lstm(x[:, i:i+1], (h_t, c_t))


            pred = self.fc1(pred)
            pred = self.lkrelu(pred)
            

            pred = pred.transpose(1, 2)
            pred = self.bn3(pred)
            pred = pred.transpose(1, 2)

            pred = self.fc2(pred)
            pred = self.lkrelu(pred)

            pred = pred.transpose(1, 2)
            pred = self.bn4(pred)
            pred = pred.transpose(1, 2)
            
            pred = self.fc3(pred)

            

            f0, l0 = torch.split(pred, 1, -1)


            x_in[:, i:i+1, 2] = f0
            x_in[:, i:i+1, 3] = l0


        e_f0, e_loudness = x_in[:, :, 2:].split(1,-1)

        return e_f0, e_loudness





        
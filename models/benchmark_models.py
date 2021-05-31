import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import numpy as np
import librosa as li



class LSTMContoursCE(nn.Module):
    
    def __init__(self, input_size = 256, hidden_size = 512, num_layers = 1):
        super(LSTMContoursCE, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(2, 64)
        self.lin2 = nn.Linear(64, input_size)

        self.lkrelu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(1999) # 2000 -1 (delay for prediction)


        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 201)





    def initialise_h0_c0(self, x):
            
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)

        return h_0, c_0



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
        out = self.lkrelu(out)
        out = self.bn(out)

        out = self.fc2(out)
        out = self.lkrelu(out)

        pitch, cents = torch.split(out, [100,101], dim = -1)

        return pitch, cents


    def pitch_cents_to_frequencies(self, pitch, cents):
    
        gen_freq = torch.tensor(li.midi_to_hz(pitch.detach().numpy())) * torch.pow(2, (cents.detach().numpy()-50)/1200)
        gen_freq = torch.unsqueeze(gen_freq, -1)

        return gen_freq



    def predict(self, pitch):
    
        f0 = torch.zeros_like(pitch)

        x = torch.cat([pitch, f0], -1)
        
        x = self.lin1(x)
        x = self.lkrelu(x)
        x = self.lin2(x)
        x = self.lkrelu(x)

        x = self.bn(x)
        
        h_t, c_t = self.initialise_h0_c0(x)

        for i in range(x.size(1)):

            pred, (h_t, c_t)  = self.lstm(x[:, i:i+1], h_t, c_t)


            out = self.lkrelu(out)

            out = self.fc1(out)
            out = self.lkrelu(out)
            out = self.bn(out)

            out = self.fc2(out)
            out = self.lkrelu(out)

            pitch, cents = torch.split(out, [100,101], dim = -1)


            f0 = self.pitch_cents_to_frequencies(pitch, cents)

            x[:, i:i+1, 1] = f0

        
        e_f0, e_loudness = x[:, :, 1:]

        return e_f0, e_loudness
        




class LSTMContoursMSE(nn.Module):
    
    def __init__(self, input_size = 256, hidden_size = 512, num_layers = 1):
        super(LSTMContoursMSE, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(2, 64)
        self.lin2 = nn.Linear(64, input_size)

        self.lkrelu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(1999) # 2000 -1 (delay for prediction)


        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 1)


    def initialise_h0_c0(self, x):
        
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)

        return h_0, c_0


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
        out = self.lkrelu(out)
        out = self.bn(out)

        out = self.fc2(out)
        out = self.lkrelu(out)

        return out

    
    def predict(self, pitch, loudness):
    
        f0 = torch.zeros_like(pitch)
        l0 = torch.zeros_like(loudness)

        x = torch.cat([pitch, loudness, f0, l0], -1)

        x = self.lin1(x)
        x = self.lkrelu(x)
        x = self.bn(x)

        x = self.lin2(x)
        x = self.lkrelu(x)
        x = self.bn(x)

        
        h_t, c_t = self.initialise_h0_c0(x)

        for i in range(x.size(1)):

            pred, (h_t, c_t)  = self.lstm(x[:, i:i+1], h_t, c_t)


            pred = self.fc1(pred)
            pred = self.lkrelu(pred)
            pred = self.bn(pred)

            pred = self.fc2(pred)
            pred = self.lkrelu(pred)
            pred = self.bn(pred)
            
            pred = self.fc3(pred)

            f0, l0 = torch.split(pred, 1, 1)

            x[:, i:i+1, 2] = f0
            x[:, i:i+1, 3] = l0

        
        e_f0, e_loudness = x[:, :, 2:]

        return e_f0, e_loudness
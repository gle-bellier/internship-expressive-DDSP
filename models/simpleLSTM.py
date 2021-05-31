
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np


class simpleLSTM(nn.Module):
    
    def __init__(self, input_size = 256, hidden_size = 512, num_layers = 1):
        super(simpleLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(2, input_size)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 2)


    def init_states(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device)

        return h_0, c_0

    def forward(self, x):
       
        x = self.lin1(x)
        h_0, c_0 = self.init_states(x)

        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        #out = out.contiguous().view(-1, self.hidden_size)

        out = self.fc1(out)

        return out

    def predict(self, x, future):

        outputs = []

        h_0, c_0 = self.init_states(x) # h_0, c_0

        x = self.lin1(x)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.fc1(out)
        
        for i in range(future):

            out = self.lin1(out)
            out, (h_n, c_n) = self.lstm(out, (h_n, c_n))
            out = self.fc1(out)

            outputs += [out]

        outputs = torch.stack(outputs, dim=2).squeeze()
        
        print("Outputs size : ", outputs.size())
        
        return outputs
            



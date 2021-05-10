
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt



class LSTMContours(nn.Module):
    def __init__(self):
        super(LSTMContours, self).__init__()
        
        self.fc1 = nn.Linear(256, 512)
        
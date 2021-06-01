import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np


class NormalizedRectifiedLinear(nn.Module):
    """
    Fused normalization / activation / linear unit
    """
    def __init__(self, input_size, output_size, activation=True, norm=True):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

        if norm:
            self.norm = nn.BatchNorm1d(output_size)
        else:
            self.norm = None

        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = nn.functional.leaky_relu(x)
        if self.norm is not None:
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)

        return x


class LSTMContours(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, num_layers=1):
        super(LSTMContours, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.pre_lstm = nn.Sequential(
            NormalizedRectifiedLinear(4, 64),
            NormalizedRectifiedLinear(64, 256),
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.post_lstm = nn.Sequential(
            NormalizedRectifiedLinear(512, 256),
            NormalizedRectifiedLinear(256, 64),
            NormalizedRectifiedLinear(64, 2, False, False),
        )

    def forward(self, x):
        x = self.pre_lstm(x)
        x = self.lstm(x)[0]
        x = self.post_lstm(x)
        return x

    def predict(self, pitch, loudness):
        f0 = torch.zeros_like(pitch)
        l0 = torch.zeros_like(loudness)

        x_in = torch.cat([pitch, loudness, f0, l0], -1)

        context = None

        for i in range(x_in.size(1)):
            x = x_in[:, i:i + 1]

            x = self.pre_lstm(x)

            pred, context = self.lstm(x, context)

            pred = self.post_lstm(pred)

            f0, l0 = torch.split(pred, 1, -1)

            x_in[:, i + 1:i + 2, 2] = f0
            x_in[:, i + 1:i + 2, 3] = l0

        e_f0, e_loudness = x_in[:, :, 2:].split(1, -1)

        return e_f0, e_loudness

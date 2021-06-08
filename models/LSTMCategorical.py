import torch
from torch.functional import norm
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributions.categorical import Categorical

import numpy as np
import librosa as li


class NormalizedRectifiedLinear(nn.Module):
    """
    Fused normalization / activation / linear unit
    """
    def __init__(self, input_size, output_size, activation=True, norm=True):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

        if norm:
            self.norm = nn.LayerNorm(output_size)
        else:
            self.norm = None

        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = nn.functional.leaky_relu(x)
        if self.norm is not None:
            # x = x.transpose(1, 2)
            x = self.norm(x)
            # x = x.transpose(1, 2)

        return x


class LSTMCategorical(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=1):
        super(LSTMCategorical, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.pre_lstm = nn.Sequential(
            NormalizedRectifiedLinear(400, 512),
            NormalizedRectifiedLinear(512, 768),
            NormalizedRectifiedLinear(768, input_size),
        )

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.post_lstm = nn.Sequential(
            NormalizedRectifiedLinear(hidden_size, 768),
            NormalizedRectifiedLinear(768, 512),
            NormalizedRectifiedLinear(512, 200),
        )

    def forward(self, x):

        x = self.pre_lstm(x)
        out = self.lstm(x)[0]
        out = self.post_lstm(out)

        cents, loudness = torch.split(out, [100, 100], dim=-1)

        return cents, loudness

    def sampling(self, cents, loudness):

        cents_dis = Categorical(logits=cents)
        loudness_dis = Categorical(logits=loudness)

        sampled_cents = cents_dis.sample().unsqueeze(-1)
        sampled_loudness = loudness_dis.sample().unsqueeze(-1)

        # need to change the range from [0, 100] -> [0, n_out]

        sampled_cents = sampled_cents.float() / 100.0
        sampled_loudness = sampled_loudness.float() / 100.0
        return sampled_cents, sampled_loudness

    def predict(self, pitch, loudness):
        f0 = torch.zeros_like(pitch)
        l0 = torch.zeros_like(loudness)

        x_in = torch.cat([pitch, loudness, f0, l0], -1)

        context = None

        for i in range(x_in.size(1) - 1):
            x = x_in[:, i:i + 1]

            x = self.pre_lstm(x)
            pred, context = self.lstm(x, context)
            pred = self.post_lstm(pred)

            e_cents, l0 = torch.split(pred, 100, -1)

            x_in[:, i + 1:i + 2, 200:300] = e_cents
            x_in[:, i + 1:i + 2, 300:] = l0

        out_cents, out_loudness = x_in[:, :, 200:].split(100, -1)
        out_cents, out_loudness = self.sampling(out_cents, out_loudness)

        return out_cents, out_loudness
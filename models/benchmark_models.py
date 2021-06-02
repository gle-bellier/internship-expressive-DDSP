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


class LSTMContoursCE(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, num_layers=1):
        super(LSTMContoursCE, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.pre_lstm = nn.Sequential(
            NormalizedRectifiedLinear(2, 64, norm=None),
            NormalizedRectifiedLinear(64, input_size),
        )

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.post_lstm = nn.Sequential(
            NormalizedRectifiedLinear(hidden_size, 256),
            NormalizedRectifiedLinear(
                256,
                201,
                norm=None,
                activation=False,
            ),
        )

    def forward(self, x):

        x = self.pre_lstm(x)
        out = self.lstm(x)[0]
        out = self.post_lstm(out)

        pitch, cents = torch.split(out, [100, 101], dim=-1)

        return pitch, cents

    def pitch_cents_to_frequencies(self, pitch, cents):

        pitch_dis = Categorical(pitch)
        cents_dis = Categorical(cents)

        sampled_pitch = pitch_dis.sample().unsqueeze(-1)
        sampled_cents = cents_dis.sample().unsqueeze(-1)

        gen_freq = torch.tensor(li.midi_to_hz(
            sampled_pitch.detach().numpy())) * torch.pow(
                2, (sampled_cents.detach().numpy() - 50) / 1200)
        gen_freq = torch.unsqueeze(gen_freq, -1)

        return gen_freq

    def predict(self, pitch):

        f0 = torch.zeros_like(pitch)

        x_in = torch.cat([pitch, f0], -1)

        context = None

        for i in range(x_in.size(1) - 1):

            x = x_in[:, i:i + 1]

            x = self.pre_lstm(x)
            pred, context = self.lstm(x, context)
            pred = self.post_lstm(pred)

            pitch, cents = torch.split(pred, [100, 101], dim=-1)
            f0 = self.pitch_cents_to_frequencies(pitch, cents)

            x[:, i + 1:i + 2, 1] = f0

        e_f0 = x[:, :, 1:]

        return e_f0


class LSTMContoursMSE(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, num_layers=1):
        super(LSTMContoursMSE, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.pre_lstm = nn.Sequential(
            NormalizedRectifiedLinear(2, 64, norm=None),
            NormalizedRectifiedLinear(64, input_size),
        )

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.post_lstm = nn.Sequential(
            NormalizedRectifiedLinear(hidden_size, 256),
            NormalizedRectifiedLinear(256, 1, norm=None),
        )

    def forward(self, x):
        x = self.pre_lstm(x)
        out = self.lstm(x)[0]
        out = self.post_lstm(out)

        return out

    def predict(self, pitch):

        f0 = torch.zeros_like(pitch)
        x_in = torch.cat([pitch, f0], -1)

        context = None

        for i in range(x_in.size(1) - 1):

            x = x_in[:, i:i + 1]
            x = self.pre_lstm(x)
            pred, context = self.lstm(x, context)
            pred = self.post_lstm(pred)

            x[:, i + 1:i + 2, 1] = pred

        e_f0 = x[:, :, 1:]

        return e_f0


class LSTMContoursFullCat(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=1):
        super(LSTMContoursFullCat, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.pre_lstm = nn.Sequential(
            NormalizedRectifiedLinear(201, 256),
            NormalizedRectifiedLinear(256, input_size),
        )

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.post_lstm = nn.Sequential(
            NormalizedRectifiedLinear(hidden_size, 256),
            NormalizedRectifiedLinear(256, 201),
        )

    def forward(self, x):

        x = self.pre_lstm(x)
        out = self.lstm(x)[0]
        out = self.post_lstm(out)

        pitch, cents = torch.split(out, [100, 101], dim=-1)

        return pitch, cents

    def pitch_cents_to_frequencies(self, pitch, cents):

        pitch_dis = Categorical(pitch)
        cents_dis = Categorical(cents)

        sampled_pitch = pitch_dis.sample().unsqueeze(-1)
        sampled_cents = cents_dis.sample().unsqueeze(-1)

        gen_freq = torch.tensor(li.midi_to_hz(
            sampled_pitch.detach().numpy())) * torch.pow(
                2, (sampled_cents.detach().numpy() - 50) / 1200)
        gen_freq = torch.unsqueeze(gen_freq, -1)

        return gen_freq
import torch
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
import librosa as li
import numpy as np


def get_data_categorical(data, n_out):

    data_idx = torch.round(data * (n_out - 1)).to(torch.int64)
    data_one_hot = F.one_hot(data_idx.squeeze(-1), num_classes=n_out)

    return data_one_hot


def get_data_quantified(data, n_out):
    """takes tensor of elt in range  [0, 1] and convert into int [0, n_out]"""

    data_idx = torch.round(data * (n_out - 1)).to(torch.int64)
    return data_idx


def get_data_from_categorical(cat, q, n_out):
    return torch.argmax(cat, dim=-1, keepdim=True) / (n_out - 1)


def frequencies_to_pitch_cents(frequencies):

    midi_pitch = li.hz_to_midi(frequencies)
    midi_pitch = np.round(midi_pitch)
    round_freq = li.midi_to_hz(midi_pitch)
    cents = (1200 * np.log2(frequencies / round_freq))

    return round_freq, cents


def pitch_cents_to_frequencies(pitch, cents):

    return pitch * torch.pow(2, cents / 1200)

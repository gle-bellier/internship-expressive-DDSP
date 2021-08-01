import torch
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
import librosa as li
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Identity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, y=None):
        return X.numpy()


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


def pctof(p, c):
    """
    convert pitch / cent to frequency
    """
    m = p + c
    return mtof(m)


def mtof(m):
    """
    converts midi note to frequency
    """
    return 440 * 2**((m - 69) / 12)


def ftopc(f):
    """
    converts frequency to pitch / cent
    """
    m_float = ftom(f)
    m_int = np.round(m_float).astype(int)
    c_float = m_float - m_int
    return m_int, c_float


def ftom(f):
    """
    converts frequency to midi note
    """
    return 12 * (np.log(f) - np.log(440)) / np.log(2) + 69
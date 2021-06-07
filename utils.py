import torch
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
import librosa as li


def get_data_categorical(data, n_out):
    data_reshaped = data.reshape(data.size(0) * data.size(1), 1)

    q = QuantileTransformer(n_quantiles=n_out - 1)
    q.fit(data_reshaped)

    data_quantile = torch.tensor(q.transform(data_reshaped))
    data_quantile = data_quantile.reshape(data.shape)

    data_idx = torch.round(data_quantile * (n_out - 1)).to(torch.int64)
    data_one_hot = F.one_hot(data_idx.squeeze(-1))

    return data_one_hot, q


def get_data_from_cat(cat, q, n_out):

    data_q = torch.argmax(cat, dim=-1, keepdim=True) / (n_out - 1)
    data_q_reshaped = data_q.reshape(data_q.size(0) * data_q.size(1), 1)
    data_reshaped = torch.tensor(q.inverse_transform(data_q_reshaped))
    data = data_reshaped.reshape(data_q.shape)
    return data


def frequencies_to_pitch_cents(frequencies):

    midi_pitch = torch.tensor(li.hz_to_midi(frequencies))
    midi_pitch = torch.round(midi_pitch).long()
    round_freq = torch.tensor(li.midi_to_hz(midi_pitch))
    cents = (1200 * torch.log2(frequencies / round_freq)).long()

    return round_freq, cents


def pitch_cents_to_frequencies(pitch, cents):

    return pitch * torch.pow(2, cents / 1200)

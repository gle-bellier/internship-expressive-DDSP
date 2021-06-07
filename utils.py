import torch
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer


def get_data_cat(data, n_out):
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
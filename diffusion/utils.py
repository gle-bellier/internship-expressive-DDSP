import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, n_dim, multiplier=30):
        super().__init__()
        self.n_dim = n_dim
        exponents = 1e-4**torch.linspace(0, 1, n_dim // 2)
        self.register_buffer("exponents", exponents)
        self.multiplier = multiplier

    def forward(self, level):
        level = level.reshape(-1, 1)
        exponents = self.exponents.unsqueeze(0)
        encoding = exponents * level * self.multiplier
        encoding = torch.stack([encoding.sin(), encoding.cos()], -1)
        encoding = encoding.reshape(*encoding.shape[:1], -1)
        return encoding.unsqueeze(-1)
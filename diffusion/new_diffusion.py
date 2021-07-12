import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class Diffusion(nn.Module):
    def __init__(self):
        self.data_range = [-1, 1]

    def set_noise_schedule(self,
                           init=torch.linspace,
                           init_kwargs={
                               "steps": 50,
                               "start": 1e-6,
                               "end": 1e-2
                           }):
        betas = init(**init_kwargs)
        alphas = 1 - betas
        alphas_cum = alphas.cumprod(dim=0)
        sqrt_alphas_cum = torch.sqrt(alphas_cum)

        sqrt_1_m_alphas_cum = torch.sqrt(1 - sqrt_alphas_cum)

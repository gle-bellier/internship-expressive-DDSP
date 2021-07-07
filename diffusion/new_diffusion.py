import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from tqdm import tqdm
from typing import Union


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_range = [-1, 1]

    @abstractmethod
    def neural_pass(self, y: torch.Tensor, cdt: torch.Tensor,
                    noise_level: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, cdt):
        raise NotImplementedError

    def set_noise_schedule(self,
                           init=torch.linspace,
                           init_kwargs={
                               "steps": 50,
                               "start": 1e-6,
                               "end": 1e-2
                           }):
        betas = init(**init_kwargs)
        alph = 1 - betas
        alph_cum = alph.cumprod(dim=0)
        alph_cum_prev_last = torch.cat([torch.tensor([1.]), alph_cum])
        alph_cum_prev = alph_cum_prev_last[:-1]
        sqrt_recip_alph_cum = (1 / alph_cum).sqrt()

        self.n_step = init_kwargs.get("steps")

        self.register_buffer("betas", betas)
        self.register_buffer("alph", alph)
        self.register_buffer("alph_cum", alph_cum)
        self.register_buffer("alph_cum_prev", alph_cum_prev)

        self.sqrt_alph_cum_prev = alph_cum_prev_last.sqrt().numpy()
        self.register_buffer("sqrt_alph_cum", alph_cum.sqrt())
        self.register_buffer("sqrt_recip_alph_cum", sqrt_recip_alph_cum)
        self.register_buffer("sqrt_1m_alph_cum", (1 - alph_cum).sqrt())

        coef1 = alph_cum_prev.sqrt() * self.betas / (1 - alph_cum)
        coef2 = self.sqrt_alph_cum * (1 - alph_cum_prev) / (1 - alph_cum)
        var = (1 - alph_cum_prev) / (1 - alph_cum) * betas
        logvar = torch.clamp(var, min=1e-20).log()

        self.register_buffer("post_mean_coef_1", coef1)
        self.register_buffer("post_mean_coef_2", coef2)
        self.register_buffer("post_logvar", logvar)

    def sample_noise_level(self, batch_size):
        s = np.random.choice(range(self.n_step), size=batch_size)
        sampled_sqrt_alph_cum = np.random.uniform(
            self.sqrt_alph_cum_prev[s],
            self.sqrt_alph_cum_prev[s + 1],
            size=batch_size,
        )
        return torch.from_numpy(sampled_sqrt_alph_cum).float()

    def compute_loss(self, y_0, cdt):
        noise_level = self.get_noise_level(y_0)
        y_noise, eps = self.diffusion_process(y_0, noise_level)
        pred_noise = self.neural_pass(y_noise, cdt, noise_level)
        loss = (eps - pred_noise).abs().mean()
        return loss

    def get_noise_level(self, y_0):
        noise_level = self.sample_noise_level(y_0.shape[0])
        noise_level = noise_level.to(y_0.device)
        noise_level = noise_level.reshape(-1, *((len(y_0.shape) - 1) * (1, )))
        return noise_level

    def diffusion_process(self, y_0, noise_level):
        """Compute n steps diffusion process
        (from y_0 to y_n) """

        # compute y_n :
        eps = torch.randn_like(y_0)
        out = noise_level * y_0 + (1 - noise_level * noise_level).sqrt() * eps

        return out, eps

    def denoising_process(self, y_n, cdt):
        y_n = torch.randn_like(y_n)
        for n in range(self.n_step)[::-1]:
            y_n = self.langevin_dynamics(y_n, cdt, n)

        #return y_0
        return y_n

    def langevin_dynamics(self, y_n, cdt, n):
        z = torch.randn_like(y_n)
        noise_level = torch.tensor(self.sqrt_alph_cum_prev[n]).to("cuda")

        # calculate y_{n-1} :
        pred_noise = self.neural_pass(y_n, cdt, noise_level)
        factor = (1. - self.alph[n]) / self.sqrt_1m_alph_cum[n]
        y = self.sqrt_recip_alph_cum[n] * (y_n - factor * pred_noise)
        eps = torch.randn_like(y) if n else torch.zeros_like(y)

        return y + eps
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from tqdm import tqdm
from typing import Union
from diffusion_model import UNet_Diffusion


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

    def q_sample(self, y_0):
        noise_level = self.sample_noise_level(y_0.shape[0])
        noise_level = noise_level.to(y_0.device)

        noise_level = noise_level.reshape(-1, *((len(y_0.shape) - 1) * (1, )))
        eps = torch.randn_like(y_0)
        out = noise_level * y_0 + (1 - noise_level * noise_level).sqrt() * eps
        return out, eps, noise_level

    def q_posterior(self, y_0, y, t):
        post_mean = self.post_mean_coef_1[t] * y_0
        post_mean += self.post_mean_coef_2[t] * y
        post_logvar = self.post_logvar[t]
        return post_mean, post_logvar

    def predict_from_noise(self, y, t, eps):
        return self.sqrt_recip_alph_cum[t] * y - self.sqrt_1m_alph_cum[t] * eps

    def p_mean_variance(self, y, cdt, t, clip=True):
        bs = y.shape[0]
        sacp = self.sqrt_alph_cum_prev[t + 1]
        noise = torch.tensor([sacp]).repeat(bs, 1).to(y)
        pred_noise = self.neural_pass(y, cdt, noise)
        y_recon = self.predict_from_noise(y, t, pred_noise)

        if clip:
            y_recon.clamp_(*self.data_range)

        model_mean, post_logvar = self.q_posterior(y_recon, y, t)
        return model_mean, post_logvar

    def inverse_dynamics(self, y, cdt, t, clip=True):
        model_mean, post_logvar = self.p_mean_variance(y, cdt, t, clip)
        eps = torch.randn_like(y) if t else torch.zeros_like(y)
        return model_mean + eps * (.5 * post_logvar).exp()

    def compute_loss(self, y, cdt):
        y_noise, eps, noise_level = self.q_sample(y)
        pred_noise = self.neural_pass(y_noise, cdt, noise_level)
        loss = (eps - pred_noise).abs().mean()
        return loss

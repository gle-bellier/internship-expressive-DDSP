import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_range = [-1, 1]

    def set_noise_schedule(self,
                           init=torch.linspace,
                           init_kwargs={
                               "steps": 50,
                               "start": 1e-6,
                               "end": 1e-2
                           }):

        self.n_step = init_kwargs.get("steps")
        betas = init(**init_kwargs)
        alphas = 1 - betas
        alphas_cum = alphas.cumprod(dim=0)
        self.sqrt_alphas_cum = torch.sqrt(alphas_cum).numpy()

        l0 = torch.tensor([1.])
        self.ls = torch.cat((l0, alphas_cum)).numpy()

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cum", alphas_cum)
        # self.register_buffer("sqrt_alpha_cum", sqrt_alphas_cum)

    def get_noise_level(self, batch_size):
        s = np.random.choice(range(self.n_step), size=batch_size)
        noise_level = np.random.uniform(
            self.ls[s],
            self.ls[s + 1],
            size=batch_size,
        )
        return torch.from_numpy(noise_level).float()

    def diffusion_process(self, y0):

        noise_level = self.get_noise_level(y0.shape[0])
        noise_level = noise_level.to(y0.device)
        eps = torch.randn_like(y0)

        noise_level = noise_level.reshape(-1, *((len(y0.shape) - 1) * (1, )))
        yn = noise_level * y0 + torch.sqrt(1 - noise_level * noise_level) * eps
        return yn, noise_level, eps

    def compute_loss(self, y0, cdt):
        yn, noise_level, eps = self.diffusion_process(y0)

        pred_eps = self.neural_pass(yn, cdt, noise_level)

        loss = torch.mean(torch.abs(eps - pred_eps))

        return loss

    def sampling(self, yn, cdt, n_steps, clip=True):
        n_step = min(n_steps, self.n_step)
        x = torch.randn_like(yn)
        for t in range(1, n_steps + 1)[::-1]:
            if t > 1:
                z = torch.randn_like(yn)
            else:
                z = torch.zeros_like(yn)

            coeff1 = 1 / self.sqrt_alphas_cum[t]
            coeff2 = self.betas[t] / (
                torch.sqrt(1 - self.alphas_cum[t] * self.alphas_cum[t]))

            noise_level = self.sqrt_alphas_cum[t]
            noise_level = torch.tensor(noise_level).to(yn.device)

            pred_eps = self.neural_pass(x, cdt, noise_level)
            sigma = self.betas[t]
            x = coeff1 * (x - coeff2 * pred_eps) + sigma * z
            if clip:
                x.clamp_(*self.data_range)

        return x

import torch
import torch.nn as nn

import pytorch_lightning as pl

from .blocks import UpBlock, FilmBlock, DownBlock
from .diffusion import DiffusionModel


class Model(pl.LightningModule, DiffusionModel):
    def __init__(self, data_dim, cdt_dim, dims):
        super().__init__()
        self.save_hyperparameters()

        # DOWN CHAIN
        ndc = [nn.Conv1d(data_dim, dims[0], 3, padding=1)]
        edc = [nn.Conv1d(cdt_dim, dims[0], 3, padding=1)]

        for i in range(len(dims) - 1):
            ndc.append(DownBlock(dims[i], dims[i + 1]))
            edc.append(DownBlock(dims[i], dims[i + 1]))

        self.ndc = nn.ModuleList(ndc)
        self.edc = nn.ModuleList(edc)

        # FILM CHAIN
        nfc = []
        efc = []

        for i in range(len(dims)):
            nfc.append(FilmBlock(dims[i], dims[i]))
            efc.append(FilmBlock(dims[i], dims[i]))

        self.nfc = nn.ModuleList(nfc)
        self.efc = nn.ModuleList(efc)

        # UP CHAIN
        dims = [dims[-1] * 2] + dims[::-1]

        uc = []

        for i in range(len(dims) - 1):
            uc.append(UpBlock(
                dims[i],
                dims[i + 1],
                dims[i + 1],
                bool(i),
            ))

        self.uc = nn.ModuleList(uc)

        self.post_process = nn.Conv1d(dims[-1], data_dim, 3, padding=1)

    def neural_pass(self, y, env, noise_level):
        ndc = []
        edc = []

        for noise_layer, envelop_layer in zip(self.ndc, self.edc):
            y = noise_layer(y)
            env = envelop_layer(env)

            ndc.append(y)
            edc.append(env)

        x = torch.cat([ndc[-1], edc[-1]], 1)

        nfc = []
        efc = []

        for noise_layer, envelop_layer in zip(self.nfc, self.efc):
            nfc.append(noise_layer(ndc.pop(0), noise_level))
            efc.append(envelop_layer(edc.pop(0), noise_level))

        for layer in self.uc:
            x = layer(x, efc.pop(-1), nfc.pop(-1))

        return self.post_process(x)

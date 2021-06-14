import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
import pytorch_lightning as pl
from newLSTMCat import FullModel
from ExpressiveDataset import ExpressiveDataset
from random import randint
from utils import *

if __name__ == "__main__":

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_total")],
        max_epochs=20000,
        resume_from_checkpoint=
        "lightning_logs/version_5/checkpoints/epoch=9810-step=147164.ckpt")
    list_transforms = [
        (Identity, ),  # u_f0 
        (MinMaxScaler, ),  # u_loudness
        (Identity, ),  # e_f0
        (Identity, ),  # e_cents
        (MinMaxScaler, ),  # e_loudness
    ]

    dataset = ExpressiveDataset(list_transforms=list_transforms)
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len
    train, val = random_split(dataset, [train_len, val_len])

    model = FullModel(360, 1024, 230, scalers=dataset.scalers)

    trainer.fit(
        model,
        DataLoader(train, 32, True),
        DataLoader(val, 32),
    )
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, random_split

torch.set_grad_enabled(False)

from newLSTMCat import FullModel
from ExpressiveDataset import ExpressiveDataset
from newLSTMpreprocess import pctof
from effortless_config import Config
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from utils import *

import json

from utils import Identity
from evaluation import Evaluator

import matplotlib.pyplot as plt


class args(Config):
    CONFIG_FILE = None


args.parse_args()
if __name__ == "__main__":

    f = open("experiments/" + args.CONFIG_FILE, )

    config = json.load(f)["config"]

    print(config)

    name = config["name"]
    logdir = config["logdir"]
    n_epochs = config["n_epochs"]

    dataset_config = config["dataset"]
    dataset_PATH = "dataset/" + dataset_config["dataset_PATH"]
    sample_length = int(dataset_config["sample_length"])
    batch_size = int(dataset_config["batch_size"])
    train_val_ratio = float(dataset_config["train_val_ratio"])

    model_config = config["model"]
    in_size = int(model_config["in_size"])
    hidden_size = int(model_config["hidden_size"])
    out_size = int(model_config["out_size"])

    list_transforms = []
    transforms_config = config["transforms"]
    types = transforms_config["types"]
    n_bins = transforms_config["n_bins"]

    for i in range(len(types)):
        if types[i] == "Identity":
            list_transforms.append((Identity, ))
        elif types[i] == "MinMaxScaler":
            list_transforms.append((MinMaxScaler, ))

        elif types[i] == "Quantile":
            if i in [1, 4]:
                list_transforms.append((QuantileTransformer, int(n_bins[0])))
            else:
                list_transforms.append((QuantileTransformer, int(n_bins[1])))

    PATH = logdir + name + "/"
    tb_logger = pl_loggers.TensorBoardLogger(PATH)
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_total")],
        max_epochs=n_epochs,
        logger=tb_logger,
    )

    dataset = ExpressiveDataset(list_transforms=list_transforms,
                                path=dataset_PATH,
                                n_sample=sample_length)

    train_len = int(train_val_ratio * len(dataset))
    val_len = len(dataset) - train_len

    train, val = random_split(dataset, [train_len, val_len])

    model = FullModel(in_size, hidden_size, out_size, scalers=dataset.scalers)

    trainer.fit(
        model,
        DataLoader(train, batch_size, True),
        DataLoader(val, batch_size),
    )

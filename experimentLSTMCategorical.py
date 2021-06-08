from tqdm import tqdm
import librosa as li

import numpy as np
import glob

from get_datasets import get_datasets
from models.LSTMCategorical import LSTMCategorical
from utils import *

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import QuantileTransformer
import signal


def save_model():
    torch.save(
        model.state_dict(),
        'results/saved_models/LSTM_Categorical_{}epochs.pt'.format(epoch))


def keyboardInterruptHandler(signal, frame):
    save_model()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)

writer = SummaryWriter("runs/benchmark/LSTMCategorical")
train_loader, test_loader, fits = get_datasets(
    dataset_file="dataset/contours.csv",
    sampling_rate=100,
    sample_duration=20,
    batch_size=16,
    ratio=0.7,
    pitch_transform="Quantile",
    loudness_transform="Quantile",
    pitch_n_quantiles=100,
    loudness_n_quantiles=100)

u_f0_fit, u_loudness_fit, e_f0_fit, e_loudness_fit, e_f0_mean_fit, e_f0_dev_fit = fits

### MODEL INSTANCIATION ###

num_epochs = 10000
learning_rate = 0.00005
loss_ratio = 0.1  # ratio between loss for pitches and loss for cents
pitch_size, cents_size = 100, 101

model = LSTMCategorical().to(device)
print("Model Classification : ")
print(model.parameters)

# Cross Entropy Loss for Classification tasks
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):

    for batch in train_loader:

        model.train()

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_dev = batch

        u_pitch_cat = get_data_categorical(u_f0, n_out=100)
        e_cents_cat = get_data_categorical(e_f0_dev, n_out=100)
        u_loudness_cat = get_data_categorical(u_loudness, n_out=100)
        e_loudness_cat = get_data_categorical(e_loudness, n_out=100)

        model_input = torch.cat([
            u_pitch_cat[:, 1:], e_cents_cat[:, :-1], u_loudness_cat[:, 1:],
            e_loudness_cat[:, :-1]
        ], -1).float()

        out_cents, out_loudness = model(model_input.to(device))
        optimizer.zero_grad()

        target_cents = get_data_quantified(
            e_f0_dev[:, 1:], n_out=100).squeeze(-1).long().to(device)
        target_loudness = get_data_quantified(
            e_loudness[:, 1:], n_out=100).squeeze(-1).long().to(device)

        out_cents = out_cents.permute(0, 2, 1).to(device)
        out_loudness = out_loudness.permute(0, 2, 1).to(device)
        # obtain the loss function
        train_loss_pitch = criterion(out_cents, target_cents)
        train_loss_cents = criterion(out_loudness, target_loudness)
        train_loss_CE = train_loss_pitch + train_loss_cents * loss_ratio

        train_loss_CE.backward()
        optimizer.step()

    # Compute validation losses :

    model.eval()
    with torch.no_grad():
        for batch in test_loader:

            u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_dev = batch

            u_pitch_cat = get_data_categorical(u_f0, n_out=100)
            e_cents_cat = get_data_categorical(e_f0_dev, n_out=100)
            u_loudness_cat = get_data_categorical(u_loudness, n_out=100)
            e_loudness_cat = get_data_categorical(e_loudness, n_out=100)

            model_input = torch.cat([
                u_pitch_cat[:, 1:], u_loudness_cat[:, 1:], e_cents_cat[:, :-1],
                e_loudness_cat[:, :-1]
            ], -1).float()

            out_cents, out_loudness = model(model_input.to(device))
            optimizer.zero_grad()

            target_cents = get_data_quantified(
                e_f0_dev[:, 1:], n_out=100).squeeze(-1).long().to(device)
            target_loudness = get_data_quantified(
                e_loudness[:, 1:], n_out=100).squeeze(-1).long().to(device)

            out_cents = out_cents.permute(0, 2, 1).to(device)
            out_loudness = out_loudness.permute(0, 2, 1).to(device)

            # TODO : Convert targets ranges

            # obtain the loss function
            test_loss_pitch = criterion(out_cents, target_cents)
            test_loss_cents = criterion(out_loudness, target_loudness)
            test_loss_CE = test_loss_pitch + test_loss_cents * loss_ratio

    if epoch % 10 == 9:
        print("Epoch: %d, training loss: %1.5f" % (epoch + 1, train_loss_CE))
        print("Epoch: %d, test loss: %1.5f" % (epoch + 1, test_loss_CE))

        writer.add_scalar('training  CEloss', train_loss_CE, epoch + 1)
        writer.add_scalar('test CEloss', test_loss_CE, epoch + 1)

torch.save(model.state_dict(),
           'results/saved_models/benchmark-CE{}epochs.pt'.format(epoch))

writer.flush()
writer.close()

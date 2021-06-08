from tqdm import tqdm
import librosa as li

import numpy as np
import glob

from get_datasets import get_datasets
from models.LSTM_towards_realistic_midi import LSTMContours
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
    torch.save(model.state_dict(),
               'results/saved_models/LSTM_MSE_CENTS_{}epochs.pt'.format(epoch))


def keyboardInterruptHandler(signal, frame):
    save_model()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)

writer = SummaryWriter("runs/benchmark/LSTM_cents_MSE")
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
learning_rate = 0.00001
loss_ratio = 0.1  # ratio between loss for pitches and loss for cents
pitch_size, cents_size = 100, 101

model = LSTMContours().to(device)
print("Model Classification : ")
print(model.parameters)

# Cross Entropy Loss for Classification tasks
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_loss(batch):
    u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_dev = batch

    # CAT CATEGORICAL INPUT
    model_input = torch.cat([
        u_f0[:, 1:],
        e_f0_dev[:, :-1],
        u_loudness[:, 1:],
        e_loudness[:, :-1],
    ], -1).float()

    # PREDICTION
    out = model(model_input.to(device))

    # COMPUTE TARGETS
    target = torch.cat([e_f0_dev[:, 1:], e_loudness[:, 1:]], -1).float()

    # COMPUTE LOSS

    loss = criterion(out, target.to(device))

    return loss


# Train the model
for epoch in range(num_epochs):
    model.train()

    mean_train_loss = 0
    nel = 0

    for batch in train_loader:

        train_loss = get_loss(batch)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        nel += 1
        mean_train_loss += (train_loss.item() - mean_train_loss) / nel

    mean_test_loss = 0
    nel = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            test_loss = get_loss(batch)

            nel += 1
            mean_test_loss += (test_loss.item() - mean_test_loss) / nel

    print("Epoch: %d, training loss: %1.5f" % (epoch + 1, mean_train_loss))
    print("Epoch: %d, test loss: %1.5f" % (epoch + 1, mean_test_loss))

    writer.add_scalar('training loss', mean_train_loss, epoch + 1)
    writer.add_scalar('test loss', mean_test_loss, epoch + 1)

torch.save(
    model.state_dict(),
    'results/saved_models/LSTM_MSE_CENTS_{}epochs.pt'.format(epoch),
)

writer.flush()
writer.close()

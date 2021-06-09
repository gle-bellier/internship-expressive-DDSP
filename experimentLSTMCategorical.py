from tqdm import tqdm
import librosa as li

import numpy as np
import glob

from get_datasets import get_datasets
from models.LSTMCategorical import LSTMCategoricalBottleneck
from utils import *

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
import signal


def save_model():
    torch.save(
        model.state_dict(),
        'results/saved_models/LSTM_Categorical_{}Bootleneck.pt'.format(epoch))


def keyboardInterruptHandler(signal, frame):
    save_model()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)

writer = SummaryWriter("runs/benchmark/LSTMCategoricalLighter")

list_transforms = [(StandardScaler, ), (QuantileTransformer, 100),
                   (MinMaxScaler, ), (QuantileTransformer, 100),
                   (MinMaxScaler, ), (MinMaxScaler, )]

train_loader, test_loader, scalers = get_datasets(
    dataset_file="dataset/contours.csv",
    transforms=list_transforms,
    sampling_rate=100,
    sample_duration=20,
    batch_size=16,
    ratio=0.7)

u_f0_fit, u_loudness_fit, e_f0_fit, e_loudness_fit, e_f0_mean_fit, e_f0_dev_fit = scalers

### MODEL INSTANCIATION ###

num_epochs = 10000
learning_rate = 0.0001
loss_ratio = 0.1  # ratio between loss for pitches and loss for cents
pitch_size, cents_size = 100, 101

model = LSTMCategoricalBottleneck().to(device)
print("Model Classification : ")
print(model.parameters)

# Cross Entropy Loss for Classification tasks
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_loss(batch):
    u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_dev = batch

    # CONTINUOUS TO CATEGORICAL
    u_pitch_cat = get_data_categorical(u_f0, n_out=100)
    e_cents_cat = get_data_categorical(e_f0_dev, n_out=100)
    u_loudness_cat = get_data_categorical(u_loudness, n_out=100)
    e_loudness_cat = get_data_categorical(e_loudness, n_out=100)

    # CAT CATEGORICAL INPUT
    model_input = torch.cat([
        u_pitch_cat[:, 1:],
        e_cents_cat[:, :-1],
        u_loudness_cat[:, 1:],
        e_loudness_cat[:, :-1],
    ], -1).float()

    # PREDICTION
    out_cents, out_loudness = model(model_input.to(device))

    # COMPUTE TARGETS
    target_cents = get_data_quantified(
        e_f0_dev[:, 1:],
        n_out=100,
    ).squeeze(-1).long().to(device)

    target_loudness = get_data_quantified(
        e_loudness[:, 1:],
        n_out=100,
    ).squeeze(-1).long().to(device)

    # COMPUTE LOSS
    out_cents = out_cents.permute(0, 2, 1).to(device)
    out_loudness = out_loudness.permute(0, 2, 1).to(device)
    train_loss_cents = criterion(out_cents, target_cents)
    train_loss_loudness = criterion(out_loudness, target_loudness)

    return train_loss_cents, train_loss_loudness


# Train the model
for epoch in range(num_epochs):
    model.train()

    mean_train_loss = 0
    mean_train_loss_cents = 0
    mean_train_loss_loudness = 0
    nel = 0

    for batch in train_loader:

        train_loss_cents, train_loss_loudness = get_loss(batch)

        train_loss_CE = train_loss_cents + train_loss_loudness
        optimizer.zero_grad()
        train_loss_CE.backward()
        optimizer.step()

        nel += 1
        mean_train_loss += (train_loss_CE.item() - mean_train_loss) / nel
        mean_train_loss_cents += (train_loss_cents.item() -
                                  mean_train_loss_cents) / nel
        mean_train_loss_loudness += (train_loss_loudness.item() -
                                     mean_train_loss_loudness) / nel

    mean_test_loss = 0
    mean_test_loss_cents = 0
    mean_test_loss_loudness = 0
    nel = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            test_loss_cents, test_loss_loudness = get_loss(batch)

            test_loss_CE = test_loss_cents + test_loss_loudness

            nel += 1
            mean_test_loss += (test_loss_CE.item() - mean_test_loss) / nel
            mean_train_loss_cents += (test_loss_cents.item() -
                                      mean_test_loss_cents) / nel
            mean_train_loss_loudness += (test_loss_loudness.item() -
                                         mean_test_loss_loudness) / nel

    print("Epoch: %d, training loss: %1.5f" % (epoch + 1, mean_train_loss))
    print("Epoch: %d, test loss: %1.5f" % (epoch + 1, mean_test_loss))

    writer.add_scalar('training CEloss', mean_train_loss, epoch + 1)
    writer.add_scalar('test CEloss', mean_test_loss, epoch + 1)

    writer.add_scalar('training loss cents', mean_train_loss_cents, epoch + 1)
    writer.add_scalar('training loss loudness', mean_train_loss_loudness,
                      epoch + 1)
    writer.add_scalar('test loss cents', mean_test_loss_cents, epoch + 1)
    writer.add_scalar('test loss loudness', mean_test_loss_loudness, epoch + 1)

torch.save(
    model.state_dict(),
    'results/saved_models/LSTM_Categorical_{}Bootleneck.pt'.format(epoch),
)

writer.flush()
writer.close()

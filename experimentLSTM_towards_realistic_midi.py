from tqdm import tqdm

import numpy as np
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.LSTM_towards_realistic_midi import LSTMContours

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler
import signal


def save_model():
    torch.save(
        model.state_dict(),
        'results/saved_models/LSTM_towards_realistic_midi{}epochs.pt'.format(
            epoch))


def keyboardInterruptHandler(signal, frame):

    save_model()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)

writer = SummaryWriter("runs/LSTM_towards_realistic_midi50000")

sc = StandardScaler()
train_loader, test_loader, fits = get_datasets(
    dataset_file="dataset/contours.csv",
    sampling_rate=100,
    sample_duration=20,
    batch_size=16,
    ratio=0.7,
    pitch_transform="Quantile",
    loudness_transform="Quantile")

u_f0_fit, u_loudness_fit, e_f0_fit, e_loudness_fit, e_f0_mean_fit, e_f0_std_fit = fits

### MODEL INSTANCIATION ###

num_epochs = 10000
learning_rate = 0.0001

model = LSTMContours().to(device)
print(model.parameters)

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

list_losses = []

# Train the model
for epoch in range(num_epochs):

    for batch in train_loader:

        model.train()

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

        model_input = torch.cat(
            [u_f0[:, 1:], u_loudness[:, 1:], e_f0[:, :-1], e_loudness[:, :-1]],
            -1).float()
        ground_truth = torch.cat([e_f0[:, 1:], e_loudness[:, 1:]], -1).float()

        output = model(model_input.to(device))
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(output, ground_truth.to(device))
        loss.backward()

        train_loss = loss.item()

        optimizer.step()

    # Compute validation loss :

    model.eval()
    with torch.no_grad():
        for batch in test_loader:

            u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

            model_input = torch.cat([
                u_f0[:, 1:], u_loudness[:, 1:], e_f0[:, :-1],
                e_loudness[:, :-1]
            ], -1)
            ground_truth = torch.cat([e_f0[:, 1:], e_loudness[:, 1:]], -1)

            output = model(model_input.to(device))
            loss = criterion(output, ground_truth.to(device))
            validation_loss = loss.item()

    if epoch % 10 == 9:

        print("Epoch: %d, training loss: %1.5f" % (epoch + 1, train_loss))
        print("Epoch: %d, validation loss: %1.5f" %
              (epoch + 1, validation_loss))

        writer.add_scalar('training loss', train_loss, epoch + 1)
        writer.add_scalar('validation loss', validation_loss, epoch + 1)

torch.save(model.state_dict(),
           'results/saved_models/LSTM_towards_realistic_midi_withpred.pt')

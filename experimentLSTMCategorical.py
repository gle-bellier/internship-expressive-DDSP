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
train_loader, test_loader, _ = get_datasets(
    dataset_file="dataset/contours.csv",
    sampling_rate=100,
    sample_duration=20,
    batch_size=16,
    ratio=0.7,
    pitch_transform=None,
    loudness_transform=None)

### MODEL INSTANCIATION ###

num_epochs = 10000
learning_rate = 0.00005
loss_ratio = 0.1  # ratio between loss for pitches and loss for cents
pitch_size, cents_size = 100, 101

model = LSTMCategorical().to(device)
print("Model Classification : ")
print(model.parameters)

criterion = torch.nn.CrossEntropyLoss(
)  # Cross Entropy Loss for Classification tasks
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):

    for batch in train_loader:

        model.train()

        u_f0, u_loudness, e_f0, e_loudness, _, _ = batch
        target_frequencies = e_f0[:, 1:]

        e_pitch, e_cents = frequencies_to_pitch_cents(e_f0)

        e_dev_cat, e_f0_fit = get_data_cat(e_f0, n_out=100)

#         u_pitch, u_cents = frequencies_to_pitch_cents(u_f0[:, 1:], 100)
#         e_pitch, e_cents = frequencies_to_pitch_cents(e_f0[:, :-1], 100)

#         model_input = torch.cat([u_f0[:, 1:], e_f0[:, :-1]], -1).float()

#         out_pitch, out_cents = model(model_input.to(device))
#         optimizer.zero_grad()

#         ground_truth_pitch, ground_truth_cents = frequencies_to_pitch_cents(
#             target_frequencies, pitch_size)

#         out_pitch = out_pitch.permute(0, 2, 1).to(device)
#         out_cents = out_cents.permute(0, 2, 1).to(device)

#         ground_truth_pitch = ground_truth_pitch.long().to(device)
#         ground_truth_cents = ground_truth_cents.long().to(device)

#         # obtain the loss function
#         train_loss_pitch = criterion(out_pitch, ground_truth_pitch)
#         train_loss_cents = criterion(out_cents, ground_truth_cents)
#         train_loss_CE = train_loss_pitch + train_loss_cents * loss_ratio

#         train_loss_CE.backward()
#         optimizer.step()

#     # Compute validation losses :

#     model.eval()
#     with torch.no_grad():
#         for batch in test_loader:

#             u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

#             u_f0 = torch.Tensor(u_f0.float())
#             e_f0 = torch.Tensor(e_f0.float())
#             target_frequencies = e_f0[:, 1:]

#             model_input = torch.cat([u_f0[:, 1:], e_f0[:, :-1]], -1)
#             model_input = std_transform(model_input)[0]

#             out_pitch, out_cents = model(model_input.to(device))

#             ground_truth_pitch, ground_truth_cents = frequencies_to_pitch_cents(
#                 target_frequencies, pitch_size)

#             # permute dimension for cross entropy loss function :

#             out_pitch = out_pitch.permute(0, 2, 1).to(device)
#             out_cents = out_cents.permute(0, 2, 1).to(device)

#             ground_truth_pitch = ground_truth_pitch.long().to(device)
#             ground_truth_cents = ground_truth_cents.long().to(device)

#             # obtain the loss function
#             test_loss_pitch = criterion(out_pitch, ground_truth_pitch)
#             test_loss_cents = criterion(out_cents, ground_truth_cents)

#             test_loss_CE = test_loss_pitch + test_loss_cents * loss_ratio

#     if epoch % 10 == 9:
#         print("Epoch: %d, training loss: %1.5f" % (epoch + 1, train_loss_CE))
#         print("Epoch: %d, test loss: %1.5f" % (epoch + 1, test_loss_CE))

#         writer.add_scalar('training  CEloss', train_loss_CE, epoch + 1)
#         writer.add_scalar('test CEloss', test_loss_CE, epoch + 1)

# torch.save(model.state_dict(),
#            'results/saved_models/benchmark-CE{}epochs.pt'.format(epoch))

# writer.flush()
# writer.close()

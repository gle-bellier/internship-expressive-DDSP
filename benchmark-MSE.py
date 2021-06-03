from tqdm import tqdm
from time import time
import librosa as li
import numpy as np
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.benchmark_models import LSTMContoursMSE

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler
import signal


def save_model():
    torch.save(model_MSE.state_dict(),
               'results/saved_models/benchmark-MSE{}epochs.pt'.format(epoch))


def keyboardInterruptHandler(signal, frame):

    save_model()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)


def std_transform(v):
    std = torch.std(v, dim=(1, 2), keepdim=True)
    m = torch.mean(v, dim=(1, 2), keepdim=True)

    return (v - m) / std, m, std


def std_inv_transform(v, m, std):
    return v * std + m


writer = SummaryWriter("runs/benchmark/MSE")

sc = StandardScaler()
train_loader, test_loader = get_datasets(dataset_file="dataset/contours.csv",
                                         sampling_rate=100,
                                         sample_duration=20,
                                         batch_size=16,
                                         ratio=0.7)

### MODEL INSTANCIATION ###

num_epochs = 4000
learning_rate = 0.0001

model_MSE = LSTMContoursMSE().to(device)
print("Model Continuous : ")
print(model_MSE.parameters)

criterion_MSE = torch.nn.MSELoss(
)  # Mean Square Error Loss for continuous contours
optimizer_MSE = torch.optim.Adam(model_MSE.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):

    for batch in train_loader:

        model_MSE.train()

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

        u_f0 = torch.Tensor(u_f0.float())
        e_f0 = torch.Tensor(e_f0.float())

        model_input = torch.cat([u_f0[:, 1:], e_f0[:, :-1]], -1)
        model_input = std_transform(model_input)[0]

        target = e_f0[:, 1:].to(device)
        out = model_MSE(model_input.to(device))

        optimizer_MSE.zero_grad()

        # obtain the loss function
        train_loss_MSE = criterion_MSE(out, target)

        train_loss_MSE.backward()
        optimizer_MSE.step()

    # Compute validation losses :

    model_MSE.eval()

    with torch.no_grad():
        for batch in test_loader:

            u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

            u_f0 = torch.Tensor(u_f0.float())
            e_f0 = torch.Tensor(e_f0.float())

            model_input = torch.cat([u_f0[:, 1:], e_f0[:, :-1]], -1)
            target = e_f0[:, 1:].to(device)

            out = model_MSE(model_input.to(device))

            # obtain the loss function
            test_loss_MSE = criterion_MSE(out, target)

    if epoch % 10 == 9:

        print("Epoch: %d, training loss: %1.5f" % (epoch + 1, train_loss_MSE))
        print("Epoch: %d, test loss: %1.5f" % (epoch + 1, test_loss_MSE))

        writer.add_scalar('training  MSEloss', train_loss_MSE, epoch + 1)
        writer.add_scalar('test MSEloss', test_loss_MSE, epoch + 1)

torch.save(model_MSE.state_dict(),
           'results/saved_models/benchmark-MSE{}epochs.pt'.format(epoch))
writer.flush()
writer.close()
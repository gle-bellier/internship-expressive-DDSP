from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.BLSTM import BLSTMContours, LSTMContours

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)

writer = SummaryWriter("runs/BLSTM")    
sc = MinMaxScaler()
train_loader, test_loader = get_datasets(dataset_file = "dataset/contours.csv", sampling_rate = 100, sample_duration = 20, batch_size = 16, ratio = 0.7, transform=sc.fit_transform)
    


### MODEL INSTANCIATION ###


num_epochs = 500
learning_rate = 0.01
input_size = 64
hidden_size = 512
num_layers = 1


model = BLSTMContours(input_size, hidden_size, num_layers).to(device)
print(model.parameters)
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)


list_losses = []


# Train the model
for epoch in range(num_epochs):
    number_of_batch = 0

    for batch in train_loader:
        number_of_batch += 1
        number_of_samples = batch[0].shape[0]
        #print("Number of samples in this batch : ", number_of_samples)

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

        u_f0 = torch.Tensor(u_f0.float())
        u_loudness = torch.Tensor(u_loudness.float())
        e_f0 = torch.Tensor(e_f0.float())
        e_loudness = torch.Tensor(e_loudness.float())
        e_f0_mean = torch.Tensor(e_f0_mean.float())
        e_f0_stddev = torch.Tensor(e_f0_stddev.float())


        ground_truth = torch.cat([e_f0, e_loudness], -1)

        
        output = model(u_f0.to(device), u_loudness.to(device))
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(output, ground_truth.to(device))
        loss.backward()
        
        optimizer.step()

    if epoch % 10 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()/number_of_batch))
        writer.add_scalar('Loss/train', loss.item()/number_of_batch , epoch)
        list_losses.append(loss.item()/number_of_batch)

writer.close()
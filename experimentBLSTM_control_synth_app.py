from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.BLSTM_Control_synth_app import BLSTMContours

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


from sklearn.preprocessing import StandardScaler

import signal

def save_model():
    torch.save(model.state_dict(), 'results/saved_models/BLSTM_control_synth{}epochs.pt'.format(epoch))

def keyboardInterruptHandler(signal, frame):

    save_model()
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using', device)



writer = SummaryWriter("runs/BSLTM_control_synth_app")    

sc = StandardScaler()
train_loader, test_loader = get_datasets(dataset_file = "dataset/contours.csv", sampling_rate = 100, sample_duration = 20, batch_size = 16, ratio = 0.7, transform=sc.fit_transform)
    


### MODEL INSTANCIATION ###


num_epochs = 10
learning_rate = 0.01
input_size = 32
hidden_size = 64
num_layers = 2


model = BLSTMContours(input_size, hidden_size, num_layers).to(device)
print(model.parameters)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


list_losses = []


# Train the model
for epoch in range(num_epochs):
    number_of_batch = 0

    for batch in train_loader:

        model.train()

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

        train_loss = loss.item() / batch[0].size(-1)
        
        optimizer.step()


    # Compute validation loss : 

    model.eval()
    with torch.no_grad():
        for batch in test_loader:

            u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

            u_f0 = torch.Tensor(u_f0.float())
            u_loudness = torch.Tensor(u_loudness.float())
            e_f0 = torch.Tensor(e_f0.float())
            e_loudness = torch.Tensor(e_loudness.float())
            e_f0_mean = torch.Tensor(e_f0_mean.float())
            e_f0_stddev = torch.Tensor(e_f0_stddev.float())


            ground_truth = torch.cat([e_f0, e_loudness], -1)
            output = model(u_f0.to(device), u_loudness.to(device))
            
            loss = criterion(output, ground_truth.to(device))
            validation_loss = loss.item() / batch[0].size(-1)


    
    if epoch % 10 == 9:

        print("Epoch: %d, training loss: %1.5f" % (epoch+1, train_loss))
        print("Epoch: %d, validation loss: %1.5f" % (epoch+1, validation_loss))

        writer.add_scalar(f'loss/check_info', {
            'training': train_loss,
            'validation': validation_loss,
            }, epoch+1)



torch.save(model.state_dict(), 'results/saved_models/LSTM_towards_realistic_midi.pt')



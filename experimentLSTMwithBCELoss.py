from tqdm import tqdm
from time import time
import librosa as li
import numpy as np
import matplotlib.pyplot as plt
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.LSTMwithBCE import LSTMContoursBCE


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


from sklearn.preprocessing import MinMaxScaler



if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)



writer = SummaryWriter("runs/LSTM_BCE")

sc = MinMaxScaler()
train_loader, test_loader = get_datasets(dataset_path = "dataset-midi-wav/", sampling_rate = 100, sample_duration = 20, batch_size = 16, ratio = 0.7, transform = None)#sc.fit_transform)
    


def frequencies_to_pitch_cents(frequencies, pitch_size, cents_size):

    # one hot vectors : 
    pitch_array = torch.zeros(frequencies.size(0), frequencies.size(1), pitch_size)
    cents_array = torch.zeros(frequencies.size(0), frequencies.size(1), cents_size)
    
    min_freq = li.midi_to_hz(0)
    max_freq = li.midi_to_hz(pitch_size-1)
    frequencies = torch.clip(frequencies, min = min_freq, max = max_freq)

    midi_pitch = torch.tensor(li.hz_to_midi(frequencies))
    midi_pitch = torch.round(midi_pitch).long()


    round_freq = torch.tensor(li.midi_to_hz(midi_pitch))
    cents = (1200 * torch.log2(frequencies / round_freq)).long()


    for i in range(0, pitch_array.size(0)):
        for j in range(0, pitch_array.size(1)):
            pitch_array[i, j, midi_pitch[i, j, 0]] = 1

    for i in range(0, pitch_array.size(0)):
        for j in range(0, pitch_array.size(1)):
            pitch_array[i, j, cents[i, j, 0] + 50] = 1


    return pitch_array, cents_array



    


### MODEL INSTANCIATION ###


num_epochs = 50
learning_rate = 0.01
input_size = 32
hidden_size = 64
num_layers = 2


model = LSTMContoursBCE(input_size, hidden_size, num_layers).to(device)
print(model.parameters)



criterion = torch.nn.BCELoss()    # mean-squared error for regression
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

        model_input = torch.squeeze(u_f0)
        model_input = torch.tensor(sc.fit_transform(model_input))
        model_input = torch.unsqueeze(model_input, -1)


        out_pitch, out_cents = model(model_input.float())
        optimizer.zero_grad()

        pitch_size, cents_size = 100, 100

 

        model_input = torch.squeeze(model_input)
        model_input = torch.tensor(sc.inverse_transform(model_input))
        model_input = torch.unsqueeze(model_input, -1)

        ground_truth_pitch, ground_truth_cents = frequencies_to_pitch_cents(model_input, pitch_size, cents_size)

        # obtain the loss function
        loss_pitch = criterion(out_pitch, ground_truth_pitch)
        loss_cents = criterion(out_pitch, ground_truth_cents)

        loss = loss_pitch + loss_cents
        loss.backward()
        
        optimizer.step()


    if epoch % 2 == 0:
        writer.add_scalar("Loss Train", loss.item()/number_of_batch, epoch)

        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()/number_of_batch))
        list_losses.append(loss.item()/number_of_batch)

writer.flush()
writer.close()


plt.plot(list_losses)
plt.title("Loss")
plt.show()




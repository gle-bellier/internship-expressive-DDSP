from tqdm import tqdm
from time import time
import librosa as li
import numpy as np
import matplotlib.pyplot as plt
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.LSTMwithNLL import LSTMContoursNLL


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



writer = SummaryWriter("runs/benchmark")

sc = MinMaxScaler()
train_loader, test_loader = get_datasets(dataset_file = "dataset/contours.csv", sampling_rate = 100, sample_duration = 20, batch_size = 16, ratio = 0.7, transform = None)#sc.fit_transform)
    


def frequencies_to_pitch_cents(frequencies, pitch_size, cents_size):
    
    # one hot vectors : 
    pitch_array = torch.zeros(frequencies.size(0), frequencies.size(1), pitch_size)
    cents_array = torch.zeros(frequencies.size(0), frequencies.size(1), cents_size)
    
    midi_pitch = torch.tensor(li.hz_to_midi(frequencies))
    midi_pitch = torch.round(midi_pitch).long()


    #print("Min =  {};  Max =  {} frequencies".format(li.midi_to_hz(0), li.midi_to_hz(pitch_size-1)))
    midi_pitch_clip = torch.clip(midi_pitch, min = 0, max = pitch_size-1)
    round_freq = torch.tensor(li.midi_to_hz(midi_pitch))
    
    cents = (1200 * torch.log2(frequencies / round_freq)).long()


    for i in range(0, pitch_array.size(0)):
        for j in range(0, pitch_array.size(1)):
            pitch_array[i, j, midi_pitch_clip[i, j, 0]] = 1

    for i in range(0, pitch_array.size(0)):
        for j in range(0, pitch_array.size(1)):
            cents_array[i, j, cents[i, j, 0] + 50] = 1


    return pitch_array, cents_array, midi_pitch_clip, cents



def pitch_cents_to_frequencies(pitch, cents):

    gen_pitch = torch.argmax(pitch, dim = -1)
    gen_cents = torch.argmax(cents, dim = -1) - 50

    gen_freq = torch.tensor(li.midi_to_hz(gen_pitch)) * torch.pow(2, gen_cents/1200)
    gen_freq = torch.unsqueeze(gen_freq, -1)

    return gen_freq


### MODEL INSTANCIATION ###


num_epochs = 50
learning_rate = 0.01
input_size = 32
hidden_size = 64
num_layers = 2

pitch_size, cents_size = 100, 100


model = LSTMContoursNLL(input_size, hidden_size, num_layers).to(device)
print(model.parameters)



criterion = torch.nn.CrossEntropyLoss()    # Cross Entropy Loss for Classification tasks
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)


list_losses = []


# Train the model
for epoch in range(num_epochs):

    for batch in train_loader:
        number_of_samples = batch[0].shape[0]

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


        out_pitch, out_cents = model(model_input.float().to(device))
        optimizer.zero_grad()

        model_input = torch.squeeze(model_input)
        model_input = torch.tensor(sc.inverse_transform(model_input))
        model_input = torch.unsqueeze(model_input, -1)

        ground_truth_pitch, ground_truth_cents = frequencies_to_pitch_cents(e_f0, pitch_size, cents_size)

        # obtain the loss function
        loss_pitch = criterion(out_pitch, ground_truth_pitch.to(device))
        loss_cents = criterion(out_pitch, ground_truth_cents.to(device))

        loss = loss_pitch + loss_cents
        loss.backward()
        
        optimizer.step()


    if epoch % 2 == 0:
        writer.add_scalar("Loss Train", loss.item(), epoch)
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        list_losses.append(loss.item())

writer.flush()
writer.close()


plt.plot(list_losses)
plt.title("Loss")
plt.show()




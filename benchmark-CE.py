from tqdm import tqdm
from time import time
import librosa as li
import numpy as np
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.benchmark_models import LSTMContoursCE


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


from sklearn.preprocessing import StandardScaler
import signal

def save_model():
    torch.save(model_CE.state_dict(), 'results/saved_models/benchmark-CE{}epochs.pt'.format(epoch))

def keyboardInterruptHandler(signal, frame):

    save_model()
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)



writer = SummaryWriter("runs/benchmark/CE")

sc = StandardScaler()
train_loader, test_loader = get_datasets(dataset_file = "dataset/contours.csv", sampling_rate = 100, sample_duration = 20, batch_size = 16, ratio = 0.7, transform = sc.fit_transform)
    


def frequencies_to_pitch_cents(frequencies, pitch_size):
    
    # one hot vectors : 
    pitch_array = torch.zeros(frequencies.size(0), frequencies.size(1))
    cents_array = torch.zeros(frequencies.size(0), frequencies.size(1))
    
    midi_pitch = torch.tensor(li.hz_to_midi(frequencies))
    midi_pitch = torch.round(midi_pitch).long()


    #print("Min =  {};  Max =  {} frequencies".format(li.midi_to_hz(0), li.midi_to_hz(pitch_size-1)))
    midi_pitch_clip = torch.clip(midi_pitch, min = 0, max = pitch_size-1)
    round_freq = torch.tensor(li.midi_to_hz(midi_pitch))
    
    cents = (1200 * torch.log2(frequencies / round_freq)).long()


    for i in range(0, pitch_array.size(0)):
        for j in range(0, pitch_array.size(1)):
            pitch_array[i, j] = midi_pitch_clip[i, j, 0]

    for i in range(0, cents_array.size(0)):
        for j in range(0, cents_array.size(1)):
            cents_array[i, j] = cents[i, j, 0] + 50


    return pitch_array, cents_array



def pitch_cents_to_frequencies(pitch, cents):

    gen_freq = torch.tensor(li.midi_to_hz(pitch)) * torch.pow(2, (cents-50)/1200)
    gen_freq = torch.unsqueeze(gen_freq, -1)

    return gen_freq


### MODEL INSTANCIATION ###


num_epochs = 1000
learning_rate = 0.01


pitch_size, cents_size = 100, 101


model_CE = LSTMContoursCE().to(device)
print("Model Classification : ")
print(model_CE.parameters)


criterion_CE = torch.nn.CrossEntropyLoss()    # Cross Entropy Loss for Classification tasks
optimizer_CE = torch.optim.Adam(model_CE.parameters(), lr=learning_rate)


# Train the model
for epoch in range(num_epochs):

    for batch in train_loader:

        model_CE.train()

        u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

        u_f0 = torch.Tensor(u_f0.float())
        e_f0 = torch.Tensor(e_f0.float())




        u_f0_in = torch.squeeze(u_f0[:,1:])
        u_f0_in = torch.tensor(sc.fit_transform(u_f0_in)).float()
        u_f0_in = torch.unsqueeze(u_f0_in, -1)
        e_f0_in = torch.squeeze(e_f0[:,1:])
        e_f0_in = torch.tensor(sc.fit_transform(e_f0_in)).float()
        e_f0_in = torch.unsqueeze(e_f0_in, -1)


        model_input = torch.cat([u_f0_in, e_f0_in], -1)

        out_pitch, out_cents = model_CE(model_input.to(device))

        optimizer_CE.zero_grad()

        target_frequencies = torch.squeeze(e_f0[:,1:])
        target_frequencies = torch.tensor(sc.inverse_transform(target_frequencies))
        target_frequencies = torch.unsqueeze(target_frequencies, -1)

        ground_truth_pitch, ground_truth_cents = frequencies_to_pitch_cents(target_frequencies, pitch_size)

        out_pitch = out_pitch.permute(0, 2, 1).to(device)
        out_cents = out_cents.permute(0, 2, 1).to(device)

        ground_truth_pitch = ground_truth_pitch.long().to(device)
        ground_truth_cents = ground_truth_cents.long().to(device)


        # obtain the loss function
        train_loss_pitch = criterion_CE(out_pitch, ground_truth_pitch)
        train_loss_cents = criterion_CE(out_cents, ground_truth_cents)
        train_loss_CE = train_loss_pitch + train_loss_cents
        
        train_loss_CE.backward()
        optimizer_CE.step()



    # Compute validation losses : 

    model_CE.eval()
    with torch.no_grad():
        for batch in test_loader:

            u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

            u_f0 = torch.Tensor(u_f0.float())
            e_f0 = torch.Tensor(e_f0.float())

            model_input = torch.cat([u_f0[:,1:], e_f0[:,:-1]], -1)

            out_pitch, out_cents = model_CE(model_input.to(device))

            target_frequencies = torch.squeeze(e_f0[:,1:])
            target_frequencies = torch.tensor(sc.inverse_transform(target_frequencies))
            target_frequencies = torch.unsqueeze(target_frequencies, -1)

            ground_truth_pitch, ground_truth_cents = frequencies_to_pitch_cents(target_frequencies, pitch_size)

            # permute dimension for cross entropy loss function :

            out_pitch = out_pitch.permute(0, 2, 1).to(device)
            out_cents = out_cents.permute(0, 2, 1).to(device)

            ground_truth_pitch = ground_truth_pitch.long().to(device)
            ground_truth_cents = ground_truth_cents.long().to(device)



            # obtain the loss function
            test_loss_pitch = criterion_CE(out_pitch, ground_truth_pitch)
            test_loss_cents = criterion_CE(out_cents, ground_truth_cents)

            test_loss_CE = test_loss_pitch + test_loss_cents

    
    if epoch % 10 == 9:
        print("--- Classification ---")
        print("Epoch: %d, training loss: %1.5f" % (epoch+1, train_loss_CE))
        print("Epoch: %d, test loss: %1.5f" % (epoch+1, test_loss_CE))

        writer.add_scalar('training  CEloss', train_loss_CE, epoch+1)
        writer.add_scalar('test CEloss', test_loss_CE, epoch+1)


torch.save(model_CE.state_dict(), 'results/saved_models/benchmark-CE{}epochs.pt'.format(epoch))

writer.flush()
writer.close()





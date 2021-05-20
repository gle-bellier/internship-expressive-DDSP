from tqdm import tqdm
from time import time
import librosa as li
import numpy as np
import matplotlib.pyplot as plt
import glob

from get_datasets import get_datasets
from get_contours import ContoursGetter
from customDataset import ContoursTrainDataset, ContoursTestDataset
from models.LSTMwithNLL import LSTMContoursNLL, LSTMContoursMSE


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


from sklearn.preprocessing import StandardScaler



if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)



writer = SummaryWriter("runs/benchmark")

sc = StandardScaler()
train_loader, test_loader = get_datasets(dataset_file = "dataset/contours.csv", sampling_rate = 100, sample_duration = 20, batch_size = 16, ratio = 0.7, transform = None)# sc.fit_transform)
    


def frequencies_to_pitch_cents(frequencies, pitch_size, cents_size):
    
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


num_epochs = 2
learning_rate = 0.01


pitch_size, cents_size = 100, 100


model_NLL = LSTMContoursNLL().to(device)
print("Model Classification : ")
print(model_NLL.parameters)

model_MSE = LSTMContoursMSE().to(device)
print("Model Continuous : ")
print(model_MSE.parameters)


criterion_NLL = torch.nn.CrossEntropyLoss()    # Cross Entropy Loss for Classification tasks
optimizer_NLL = torch.optim.Adam(model_NLL.parameters(), lr=learning_rate)

criterion_MSE = torch.nn.MSELoss()    # Mean Square Error Loss for continuous contours
optimizer_MSE = torch.optim.Adam(model_MSE.parameters(), lr=learning_rate)

#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)


# Train the model
for epoch in range(num_epochs):

    for batch in train_loader:

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

        out_pitch, out_cents = model_NLL(model_input.to(device))
        out_continuous = model_MSE(model_input.to(device))

        optimizer_NLL.zero_grad()
        optimizer_MSE.zero_grad()

        target_frequencies = torch.squeeze(e_f0[:,1:])
        target_frequencies = torch.tensor(sc.inverse_transform(target_frequencies))
        target_frequencies = torch.unsqueeze(target_frequencies, -1)

        ground_truth_pitch, ground_truth_cents = frequencies_to_pitch_cents(e_f0[:,1:], pitch_size, cents_size)

        print("Ground Truth size {}, Model out size {}".format(ground_truth_pitch.size(), out_pitch.size()))

        out_pitch = out_pitch.permute(0, 2, 1)
        out_cents = out_cents.permute(0, 2, 1)

        # ground_truth_pitch = ground_truth_pitch.permute(0, 2, 1)
        # ground_truth_cents = ground_truth_cents.permute(0, 2, 1)

        ground_truth_pitch = ground_truth_pitch.long()
        ground_truth_cents = ground_truth_cents.long()


        print("Ground Truth size {}, Model out size {}".format(ground_truth_pitch.size(), out_pitch.size()))

        # obtain the loss function
        train_loss_pitch = criterion_NLL(out_pitch, ground_truth_pitch.to(device))
        train_loss_cents = criterion_NLL(out_cents, ground_truth_cents.to(device))
        train_loss_MSE = criterion_MSE(out_continuous, e_f0[:,1:])
        train_loss_NLL = train_loss_pitch + train_loss_cents
        
        train_loss_NLL.backward()
        train_loss_MSE.backward()
        
        optimizer_NLL.step()
        optimizer_MSE.step()



    # Compute validation losses : 

    model_MSE.eval()
    model_NLL.eval()
    with torch.no_grad():
        for batch in test_loader:

            u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev = batch

            u_f0 = torch.Tensor(u_f0.float())
            e_f0 = torch.Tensor(e_f0.float())

            model_input = torch.cat([u_f0[:,1:], e_f0[:,:-1]], -1)

            out_pitch, out_cents = model_NLL(model_input.to(device))
            out_continuous = model_MSE(model_input.to(device))

            target_frequencies = torch.squeeze(e_f0[:,1:])
            target_frequencies = torch.tensor(sc.inverse_transform(target_frequencies))
            target_frequencies = torch.unsqueeze(target_frequencies, -1)

            ground_truth_pitch, ground_truth_cents = frequencies_to_pitch_cents(e_f0, pitch_size, cents_size)

            # permute dimension for cross entropy loss function :

            out_pitch = out_pitch.permute(0, 2, 1)
            out_cents = out_cents.permute(0, 2, 1)

            ground_truth_pitch = ground_truth_pitch.permute(0, 2, 1)
            ground_truth_cents = ground_truth_cents.permute(0, 2, 1)

            # obtain the loss function
            test_loss_pitch = criterion_NLL(out_pitch, ground_truth_pitch.to(device))
            test_loss_cents = criterion_NLL(out_pitch, ground_truth_cents.to(device))
            test_loss_MSE = criterion_MSE(out_continuous, e_f0[:,1:])

            test_loss_NLL = test_loss_pitch + test_loss_cents

    
    if epoch % 10 == 9:
        print("--- Classification ---")
        print("Epoch: %d, training loss: %1.5f" % (epoch+1, train_loss_NLL))
        print("Epoch: %d, test loss: %1.5f" % (epoch+1, test_loss_NLL))

        print("--- Continuous ---")
        print("Epoch: %d, training loss: %1.5f" % (epoch+1, train_loss_MSE))
        print("Epoch: %d, test loss: %1.5f" % (epoch+1, test_loss_MSE))

        writer.add_scalar('training  NLLloss', train_loss_NLL, epoch+1)
        writer.add_scalar('test NLLloss', test_loss_NLL, epoch+1)

        writer.add_scalar('training  MSEloss', train_loss_MSE, epoch+1)
        writer.add_scalar('test MSEloss', test_loss_MSE, epoch+1)



writer.flush()
writer.close()





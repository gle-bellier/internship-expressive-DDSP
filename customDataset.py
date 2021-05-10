import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ContoursTrainDataset(Dataset):
    """ Unexpressive and expressive contours Dataset"""


    def __init__(self, u_f0, u_loudness, e_f0, e_loudness, sample_length=2000, overlap = 0.3, transform=None):

        self.transform = transform
        self.sample_length = sample_length
        self.overlap = overlap

        self.u_f0 = u_f0
        self.u_loudness = u_loudness
        self.e_f0 = e_f0
        self.e_loudness = e_loudness

        self.length = len(self.u_f0)    


        self.segments = []



    def __len__(self):
        return self.length//self.sample_length


    def get_random_indexes(self):

        seg_length = int((1 - self.overlap) * self.sample_length)
        i_max = np.floor((self.length - self.sample_length)/seg_length)

        i = np.random.randint(i_max+1)
        return int(i*seg_length), int(i*seg_length + self.sample_length)



    def __getitem__(self, idx):
        start, end = self.get_random_indexes()
        #print("Indexes: [{}:{}]".format(start, end))
        self.segments.append((start,end))
        return self.u_f0[start:end], self.u_loudness[start:end], self.e_f0[start:end], self.e_loudness[start:end]

        
class ContoursTestDataset(Dataset):
    """ Unexpressive and expressive contours Dataset"""


    def __init__(self, u_f0, u_loudness, e_f0, e_loudness, sample_length=2000, transform=None):

        self.transform = transform
        self.sample_length = sample_length

        self.u_f0 = u_f0
        self.u_loudness = u_loudness
        self.e_f0 = e_f0
        self.e_loudness = e_loudness

        self.length = len(self.u_f0)

        self.segments = []


    def __len__(self):
        return self.length//self.sample_length
        



    def __getitem__(self, idx):
        start = int(idx * self.sample_length)
        end = int(start + self.sample_length)
        self.segments.append((start,end))

        #print("Indexes: [{}:{}]".format(start, end))
        return self.u_f0[start:end], self.u_loudness[start:end], self.e_f0[start:end], self.e_loudness[start:end]
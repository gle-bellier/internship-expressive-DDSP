import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ContoursTrainDataset(Dataset):
    """ Unexpressive and expressive contours Dataset"""


    def __init__(self, u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev, seq_length = 20, sample_length=2000, overlap = 0.3, transform=None):

        self.transform = transform
        self.sample_length = sample_length
        self.overlap = overlap
        self.seq_length = seq_length

        self.u_f0 = u_f0
        self.u_loudness = u_loudness
        self.e_f0 = e_f0
        self.e_loudness = e_loudness
        self.e_f0_mean = e_f0_mean
        self.e_f0_stddev = e_f0_stddev

        if self.transform is not None:
            self.u_f0 = self.transform(self.u_f0.reshape(-1,1))
            self.u_loudness = self.transform(self.u_loudness.reshape(-1,1))
            self.e_f0 = self.transform(self.e_f0.reshape(-1,1))
            self.e_loudness = self.transform(self.e_loudness.reshape(-1,1))
            self.e_f0_mean = self.transform(self.e_f0_mean.reshape(-1,1))
            self.e_f0_stddev = self.transform(self.e_f0_stddev.reshape(-1,1))

        self.length = len(self.u_f0)    
        self.segments = []
        
        print("ufqdb =")
        print(self.u_f0.shape)


    def __len__(self):
        return int(self.length/((1-self.overlap)*self.sample_length))


    def get_random_indexes(self):

        seg_length = int((1 - self.overlap) * self.sample_length)
        i_max = np.floor((self.length - self.sample_length)/seg_length)

        i = np.random.randint(i_max+1)
        return int(i*seg_length), int(i*seg_length + self.sample_length)



    def sliding_windows_freq(self, u_f0, e_f0, e_f0_mean, e_f0_stddev): # label like
        x, y , y_mean, y_stddev = [], [], [], []
        for i in range(len(u_f0)-self.seq_length-1):
            _x = u_f0[i:(i+self.seq_length)]
            _y = e_f0[i+self.seq_length]
            _y_mean = e_f0_mean[i+self.seq_length]
            _y_stddev = e_f0_stddev[i+self.seq_length]
            x.append(_x)
            y.append(_y)
            y_mean.append(_y_mean)
            y_stddev.append(_y_stddev)

        return np.array(x), np.array(y), np.array(y_mean), np.array(y_stddev)

    def sliding_windows_loudness(self, u_loudness, e_loudness): # label like
        x = []
        y = []
        for i in range(len(u_loudness)-self.seq_length-1):
            _x = u_loudness[i:(i+self.seq_length)]
            _y = e_loudness[i+self.seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x),np.array(y)


    def __getitem__(self, idx):
        start, end = self.get_random_indexes()
        #print("Indexes: [{}:{}]".format(start, end))
        self.segments.append((start,end))


        # get windows : 

        u_f0, e_f0, e_f0_mean, e_f0_stddev = self.sliding_windows_freq(self.u_f0[start:end], self.e_f0[start:end], self.e_f0_mean[start:end], self.e_f0_stddev[start:end])
        u_loudness, e_loudness = self.sliding_windows_loudness(self.u_loudness[start:end], self.e_loudness[start:end])

        return u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev
        




class ContoursTestDataset(Dataset):
    """ Unexpressive and expressive contours Dataset"""


    def __init__(self, u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev, seq_length = 20, sample_length=2000, overlap = 0.3, transform=None):

        self.transform = transform
        self.sample_length = sample_length
        self.overlap = overlap
        self.seq_length = seq_length

        self.u_f0 = u_f0
        self.u_loudness = u_loudness
        self.e_f0 = e_f0
        self.e_loudness = e_loudness
        self.e_f0_mean = e_f0_mean
        self.e_f0_stddev = e_f0_stddev

        if self.transform is not None:
            self.u_f0 = self.transform(self.u_f0.reshape(-1,1))
            self.u_loudness = self.transform(self.u_loudness.reshape(-1,1))
            self.e_f0 = self.transform(self.e_f0)
            self.e_loudness = self.transform(self.e_loudness.reshape(-1,1))
            self.e_f0_mean = self.transform(self.e_f0_mean.reshape(-1,1))
            self.e_f0_stddev = self.transform(self.e_f0_stddev.reshape(-1,1))
        


        self.length = len(self.u_f0)    
        self.segments = []


    def get_random_indexes(self):

        seg_length = int((1 - self.overlap) * self.sample_length)
        i_max = np.floor((self.length - self.sample_length)/seg_length)

        i = np.random.randint(i_max+1)
        return int(i*seg_length), int(i*seg_length + self.sample_length)



    def sliding_windows_freq(self, u_f0, e_f0, e_f0_mean, e_f0_stddev): # label like
        x, y , y_mean, y_stddev = [], [], [], []
        for i in range(len(u_f0)-self.seq_length-1):
            _x = u_f0[i:(i+self.seq_length)]
            _y = e_f0[i+self.seq_length]
            _y_mean = e_f0_mean[i+self.seq_length]
            _y_stddev = e_f0_stddev[i+self.seq_length]
            x.append(_x)
            y.append(_y)
            y_mean.append(_y_mean)
            y_stddev.append(_y_stddev)

        return np.array(x), np.array(y), np.array(y_mean), np.array(y_stddev)

    def sliding_windows_loudness(self, u_loudness, e_loudness): # label like
        x = []
        y = []
        for i in range(len(u_loudness)-self.seq_length-1):
            _x = u_loudness[i:(i+self.seq_length)]
            _y = e_loudness[i+self.seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x),np.array(y)




    def __len__(self):
        return self.length//self.sample_length
        

    def __getitem__(self, idx):
        start = int(idx * self.sample_length)
        end = int(start + self.sample_length)
        self.segments.append((start,end))

        # get windows : 

        u_f0, e_f0, e_f0_mean, e_f0_stddev = self.sliding_windows_freq(self.u_f0[start:end], self.e_f0[start:end], self.e_f0_mean[start:end], self.e_f0_stddev[start:end])
        u_loudness, e_loudness = self.sliding_windows_loudness(self.u_loudness[start:end], self.e_loudness[start:end])

        return u_f0, u_loudness, e_f0, e_loudness, e_f0_mean, e_f0_stddev
        






















import torch
import pickle
from torch.utils.data import Dataset


class FrequencyTrainDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.labels = pickle.load(open('train_targets.pkl', 'rb'))
        self.freqs = pickle.load(open('freq_train_data.pkl', 'rb'))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        freq = self.freqs[idx]
        freq = torch.Tensor(freq).reshape(1, 3000).double()
        label = [self.labels[idx]]
        label = torch.Tensor(label).double()
        if self.transform:
            freq = self.transform(freq)
        if self.target_transform:
            label = self.target_transform(label)
        return freq, label


class FrequencyTestDataset(Dataset):
    def __init__(self, transform=None):
        self.files = pickle.load(open('test_file.pkl', 'rb'))
        self.freqs = pickle.load(open('freq_test_data.pkl', 'rb'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        freq = self.freqs[idx]
        files = self.files[idx]
        freq = torch.Tensor(freq).reshape(1, 3000).double()
        if self.transform:
            freq = self.transform(freq)
        return freq, files

# data = FrequencyTestDataset()
# print(data[0])
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Create the plot
# plt.plot(data[0][0][0])
# plt.show()
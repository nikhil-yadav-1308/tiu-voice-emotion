import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import config
config = config()

class CustomDataset(Dataset):
    def __init__(self, spectrogram_csv, spectrogram_dir):
        """
        spectrogram_csv: csv where each row contains the path name to a spectrogram and its label
        spectrogram_dir: directory where the spectrograms are stored 
        """
        self.spectrogram_csv = spectrogram_csv
        self.spectrogram_dir = spectrogram_dir
        self.len = len(spectrogram_csv)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Function that returns a spectrogram and its label
        # __getitem__ is called by the DataLoader object that subsumes this Dataset in batches of batch_size
        spectrogram_path = self.spectrogram_csv.iloc[idx, 0]
        label = self.spectrogram_csv.iloc[idx, 1]
        spectrogram = np.load(spectrogram_path)
        spectrogram = torch.from_numpy(spectrogram)
        return spectrogram, label
        
def data_generator(spectrogram_csv_path, spectrogram_dir, test_csv_path=False):
    """
    Returns DataLoader objects for train and val data, unless a test_csv_path is specified, 
    in which case a third DataLoader for the test set is returned. 
    """
    spectrogram_csv = pd.read_csv(spectrogram_csv_path)

    # Partition samples into training and validation sets
    train_csv, val_csv = train_test_split(spectrogram_csv, test_size=config.val_size, random_state=config.seed)
    
    # Only return test_dataloader if a test_csv_path is given
    if test_csv_path:
        # Read test csv
        test_csv = pd.read(test_csv_path)

        # Create custom Dataset objects that are then fed to the DataLoader
        train_data = CustomDataset(train_csv, spectrogram_dir)
        val_data = CustomDataset(val_csv, spectrogram_dir)
        test_data = CustomDataset(test_csv, spectrogram_dir)

        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=config.drop_last)
        val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, drop_last=config.drop_last)
        test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, drop_last=config.drop_last)
        return train_dataloader, val_dataloader, test_dataloader
    else:
        train_data = CustomDataset(train_csv, spectrogram_dir)
        val_data = CustomDataset(val_csv, spectrogram_dir)

        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=config.drop_last)
        val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, drop_last=config.drop_last)
        return train_dataloader, val_dataloader
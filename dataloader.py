import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, spectrogram_csv, spectogram_dir):
        """
        csv: a csv file where each row contains the name of the image file and its label
        array_dir: the directory in which the spectrogram arrays are stored
        """
        self.spectrogram_csv = spectrogram_csv
        self.spectogram_dir = spectogram_dir
        self.len = len(spectrogram_csv)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns a spectogram and its label.
        """
        spectrogram_path = self.spectrogram_csv.iloc[idx, 0]
        label = self.spectrogram_csv.iloc[idx, 1]
        spectrogram = np.load(spectrogram_path)
        spectrogram = torch.from_numpy(spectrogram)
        return spectrogram, label
        
def data_generator(spectrogram_csv_path, spectogram_dir, config, test_csv_path=False):
    """
    Returns DataLoader objects for train and val data, unless a test_csv_path is specified, 
    in which case an additional DataLoader for the test set is returned. 
    """
    # Load the csv
    spectrogram_csv = pd.read_csv(spectrogram_csv_path)

    # Partition samples into training and validation sets
    train_csv, val_csv = train_test_split(spectrogram_csv, test_size=config.val_size, random_state=config.seed)
    print(f"Partioned data into train and val\ntraining set size: {len(train_csv)}\tvalidation set size: {len(val_csv)}")
    
    # Only return test_dataloader if a test_csv_path is given
    if test_csv_path:
        # Read test csv
        test_csv = pd.read(test_csv_path)

        train_data = CustomDataset(train_csv, spectogram_dir)
        val_data = CustomDataset(val_csv, spectogram_dir)
        test_data = CustomDataset(test_csv, spectogram_dir)

        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=config.drop_last)
        val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, drop_last=config.drop_last)
        test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, drop_last=config.drop_last)
        print(f"Returning 3 dataloaders")
        return train_dataloader, val_dataloader, test_dataloader
    else:
        train_data = CustomDataset(train_csv, spectogram_dir)
        val_data = CustomDataset(val_csv, spectogram_dir)

        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=config.drop_last)
        val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, drop_last=config.drop_last)
        print(f"Returning 2 dataloaders")
        return train_dataloader, val_dataloader
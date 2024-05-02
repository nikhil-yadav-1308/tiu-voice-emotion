import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, spectrogram_csv, array_dir):
        """
        csv: a csv file where each row contains the name of the image file and its label
        array_dir: the directory in which the spectrogram arrays are stored
        """
        self.spectrogram_csv = spectrogram_csv
        self.array_dir = array_dir
        self.len = len(self.img_labels)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns an array (a spectogram) and its label.
        First attempts to read the array as a numpy array, then pytorch tensor.
            If it successfully reads the data as numpy array, it converts it into a tensor.
        """
        array_path = os.path.join(self.array_dir, self.spectrogram_csv.iloc[idx, 0])
        # Try reading data as numpy arrays
        try:
            array = np.load(array_path)
            array = torch.from_numpy(array)
            label = self.spectrogram_csv.iloc[idx, 1]
            return array, label
        
        except Exception as e:
            print(f"Exception occurred while reading data as numpy arrays: {e}\nAttempting to read data as tensors")            
            try:
                array = torch.load(array_path)
                label = self.spectrogram_csv.iloc[idx, 1]
                return array, label
            
            except Exception as e:
                raise AssertionError(f"Exception occurred while reading data as tensors: {e}")

def data_generator(train_csv_path, array_dir, config, test_csv_path=False):
    """
    Returns DataLoaders for training and validation data, unless a test_csv_path is specified, 
    in which case an additional DataLoader for test is returned. 
    """
    annotations_csv = pd.read_csv(train_csv_path)

    # Partition samples into training and validation sets
    train_csv, val_csv = train_test_split(annotations_csv, test_size=config.val_size)

    # Only return test_dataloader if a test_csv_path is given
    if test_csv_path:
        train_data = CustomDataset(train_csv, array_dir)
        val_data = CustomDataset(val_csv, array_dir)
        test_csv = pd.read(test_csv_path)
        test_data = CustomDataset(test_csv, array_dir)

        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=config.drop_last)
        val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, drop_last=config.drop_last)
        test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, drop_last=config.drop_last)

        return train_dataloader, val_dataloader, test_dataloader
    else:
        train_data = CustomDataset(train_csv, array_dir)
        val_data = CustomDataset(val_csv, array_dir)

        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=config.drop_last)
        val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, drop_last=config.drop_last)

        return train_dataloader, val_dataloader
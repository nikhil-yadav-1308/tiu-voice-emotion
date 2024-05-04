import os
import torch
import numpy as np
import pandas as pd

from config import config
from data_preprocessing import spectograms_with_csv
from dataloader import data_generator

config = config()

# Call spectograms_with_csv to create the spectrograms and csv
input_dir = os.path.join("train_data", "train_subset")
output_dir = "mel_spectograms"
csv_file_path = "mel_spectograms.csv"

#### UNCOMMENT THIS TO CREATE SPECTOGRAMS AND ITS ACCOMPANYING CSV FILE ####
# spectograms_with_csv(input_dir=input_dir, 
#                      output_dir=output_dir, 
#                      csv_file_path=csv_file_path)

# Create training and validation sets
train_dataloader, val_dataloader = data_generator(spectrogram_csv_path=csv_file_path, 
                                                  spectogram_dir=output_dir,
                                                  config=config)

# Check if the shit returned is correct
train_spectogram, train_label = next(iter(train_dataloader))
print(type(train_spectogram), type(train_label))
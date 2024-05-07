import os
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd

from config import config
config = config()

from data_preprocessing import spectograms_with_csv
from dataloader import data_generator
from trainer import trainer

# Set random seeds
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# Call spectograms_with_csv to create the spectrograms and csv
input_dir = os.path.join("train_data", "train_subset")  # Where to find raw audio files
output_dir = "mel_spectograms"                          # Where to store padded spectograms
csv_path = "mel_spectograms.csv"                        # Name of the CSV file that holds spectogram path and label

# ### UNCOMMENT THIS TO CREATE SPECTOGRAMS AND ITS ACCOMPANYING CSV FILE ####
# spectograms_with_csv(input_dir=input_dir, 
#                      output_dir=output_dir, 
#                      csv_path=csv_path)

# Create training and validation sets
train_dataloader, val_dataloader = data_generator(spectrogram_csv_path=csv_path, 
                                                  spectogram_dir=output_dir,
                                                  config=config)

trainer(train_dataloader)
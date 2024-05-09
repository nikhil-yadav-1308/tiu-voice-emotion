import os
import torch
import numpy as np

from data_preprocessing import spectrograms_with_csv
from dataloader import data_generator
from trainer import trainer

from config import config
config = config()


# Set random seeds
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# Call spectrograms_with_csv to create the padded spectrograms and csv
input_dir = os.path.join("train_data", "train_subset")  # Directory to raw audio files
output_dir = "mel_spectrograms"                         # Directory to store padded spectrograms
csv_path = "mel_spectrograms.csv"                       # Path name to CSV file that holds paths and label

# # ### UNCOMMENT THIS TO CREATE PADDED SPECTROGRAMS AND ITS ACCOMPANYING CSV FILE ####

# spectrograms_with_csv(input_dir=input_dir, 
#                      output_dir=output_dir, 
#                      csv_path=csv_path)

# Create training and validation sets
train_dataloader, val_dataloader = data_generator(spectrogram_csv_path=csv_path, spectrogram_dir=output_dir)

# Train and evaluate the model
trainer(train_dataloader, val_dataloader)
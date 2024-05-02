import pickle
import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm

# Function to load pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load data from directory
dir = 'train_subset'
print("Loading data from directory: ", dir)

# Function to convert audio to Mel spectrogram and save as a NumPy binary file (.npy)
def audio_to_melspectrogram(audio, file_path):
    # Parameters for the spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=8000, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    # Save the spectrogram as a NumPy binary file (.npy)
    np.save(file_path, log_S)

# Prepare to save spectrograms and paths/targets for CSV
os.makedirs('spectrograms', exist_ok=True)
paths = []
targets = []

# Process each file immediately as it is loaded
shortest = 1000000
longest = 0
for file in tqdm(os.listdir(dir)):
    if file.endswith('.pkl'):
        file_path = os.path.join(dir, file)
        file_contents = load_pickle(file_path)
        valence = file_contents['valence']
        audio_data = file_contents['audio_data']

        # Construct the path for saving the spectrogram
        spectrogram_path = os.path.join('spectrograms', f'spectrogram_{len(paths)}.npy')
        audio_to_melspectrogram(audio_data, spectrogram_path)
        if len(audio_data) < shortest:
            shortest = len(audio_data)
        if len(audio_data) > longest:
            longest = len(audio_data)
        # Append the path and target to lists
        paths.append(spectrogram_path)
        targets.append(valence)

print("Shortest audio: ", shortest, " samples, corresponding to ", shortest/8000, " seconds")
print("Longest audio: ", longest, " samples, corresponding to ", longest/8000, " seconds")

# Save paths and targets to a CSV file
data_df = pd.DataFrame({
    'path': paths,
    'valence': targets
})
data_df.to_csv('preprocessed_data.csv', index=False)

print("Finished processing and saving data.")
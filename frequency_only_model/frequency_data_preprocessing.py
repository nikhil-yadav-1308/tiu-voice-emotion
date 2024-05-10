import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.fft import rfft


# Function to load pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


# Load data from directory
train_dir = '../train/'
test_dir = '../test/'
freq_train_data = []
freq_test_data = []
train_targets = []
test_file = []

# Train data
shortest = np.Inf
longest = 0
for file in tqdm(os.listdir(train_dir)):
    if file.endswith('.pkl'):
        file_path = os.path.join(train_dir, file)
        file_contents = load_pickle(file_path)
        valence = file_contents['valence']
        audio_data = file_contents['audio_data']
        values = rfft(audio_data)[:3000]
        if len(audio_data) < 6000:
            print(len(audio_data))
            continue
        values = np.array(list(map(np.real, values)))
        freq_train_data.append(values)
        if len(audio_data) < shortest:
            shortest = len(audio_data)
        if len(audio_data) > longest:
            longest = len(audio_data)
        train_targets.append(valence)

print("Shortest audio: ", shortest, " samples, corresponding to ", shortest / 8000, " seconds")
print("Longest audio: ", longest, " samples, corresponding to ", longest / 8000, " seconds")

# Test data
shortest = np.Inf
longest = 0
for file in tqdm(os.listdir(test_dir)):
    if file.endswith('.pkl'):
        file_path = os.path.join(test_dir, file)
        file_contents = load_pickle(file_path)
        audio_data = file_contents['audio_data']
        values = rfft(audio_data)[:3000]
        if len(audio_data) < 6000:
            print(len(audio_data))
        values = np.array(list(map(np.real, values)))
        freq_test_data.append(values)
        if len(audio_data) < shortest:
            shortest = len(audio_data)
        if len(audio_data) > longest:
            longest = len(audio_data)
        test_file.append(file)

print("Shortest audio: ", shortest, " samples, corresponding to ", shortest / 8000, " seconds")
print("Longest audio: ", longest, " samples, corresponding to ", longest / 8000, " seconds")

freq_train_data = np.array(freq_train_data)
freq_test_data = np.array(freq_test_data)
train_targets = np.array(train_targets)
test_file = np.array(test_file)

pickle.dump(freq_train_data, open('freq_train_data.pkl', "wb"))
pickle.dump(freq_test_data, open('freq_test_data.pkl', "wb"))
pickle.dump(train_targets, open('train_targets.pkl', "wb"))
pickle.dump(test_file, open('test_file.pkl', "wb"))

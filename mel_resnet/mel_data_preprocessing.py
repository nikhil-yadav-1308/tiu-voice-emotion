import pickle
import numpy as np
import pandas as pd
import os
from librosa.feature import melspectrogram
from librosa import power_to_db
from tqdm import tqdm
from mel_config import MelConfig
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

config = MelConfig()


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def audio_to_melspectrogram(audio, file_path, longest_sequence):
    S = melspectrogram(y=audio, sr=8000, n_mels=config.num_mels, hop_length=config.hop_length)
    log_S = power_to_db(S, ref=np.max)
    # print(spectrogram(audio, mode='phase'))
    # frequencies, times, phase = spectrogram(audio, fs=8000, nperseg=510, noverlap=200, mode='phase')

    # print(S.shape)
    # print(phase.shape)
    # print(log_S.shape)
    # print(np.concatenate((log_S, phase), axis=1).shape)
    # log_S = np.concatenate((log_S, power_to_db(phase, ref=np.max)), axis=1)

    # plt.figure(figsize=(10, 8))
    # plt.imshow(log_S, aspect='auto', origin='lower', extent=[0, S.shape[1]*config.hop_length/8000, 0, S.shape[0]])
    # plt.title('Mel Spectrogram')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Frequency')
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    # plt.show()

    # Save the spectrogram as a NumPy binary file (.npy)
    np.save(file_path, log_S)
    # Keep track of the longest sequence, will be used to pad the spectrograms
    if log_S.shape[-1] >= longest_sequence:
        longest_sequence = log_S.shape[-1]
    return longest_sequence


def pad_spectrograms(spectrogram_path, longest_sequence):
    spectrogram = np.load(spectrogram_path)
    # Calculate how much padding is desired
    padding_amount = longest_sequence - spectrogram.shape[1]
    # Append 0s to the spectrogram array along axis 1
    # print(np.mean(spectrogram))
    padded_spectrogram = np.append(spectrogram, np.zeros(shape=(spectrogram.shape[0], padding_amount)) + np.mean(spectrogram), axis=1)
    # Ensure that the padded sequence has the correct dimensions
    assert padded_spectrogram.shape[1] == longest_sequence
    np.save(spectrogram_path, padded_spectrogram)


def spectrograms_with_csv(input_dir, output_dir, csv_path="spectrogram.csv"):
    """
    input_dir: directory to the raw audio files
    output_dir: directory to save the generated spectrograms
    csv_path: path name of the csv file
    """
    # Load data from input directory
    print(f"Loading audio data from directory: {input_dir}")

    # Prepare to save spectrograms and paths/targets for CSV
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating spectrograms and saving to {output_dir}...")
    paths = []
    targets = []
    longest_sequence = 0

    # Process each file immediately as it is loaded
    shortest = 1000000
    longest = 0
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith('.pkl'):
            file_path = os.path.join(input_dir, file)
            file_contents = load_pickle(file_path)
            valence = file_contents['valence']
            audio_data = file_contents['audio_data']

            # Construct the path for saving the spectrogram
            spectrogram_path = os.path.join(output_dir, f'spectrogram_{len(paths)}.npy')
            longest_sequence = audio_to_melspectrogram(audio_data, spectrogram_path, longest_sequence)
            # return
            if len(audio_data) < shortest:
                shortest = len(audio_data)
            if len(audio_data) > longest:
                longest = len(audio_data)
            # Append the path and target to lists
            paths.append(spectrogram_path)
            targets.append(valence)

    print(f"Padding spectrograms according to the longest sequence: {longest_sequence}...")
    for spectrogram_path in paths:
        pad_spectrograms(spectrogram_path, longest_sequence)

    print("Shortest audio: ", shortest, " samples, corresponding to ", shortest / 8000, " seconds")
    print("Longest audio: ", longest, " samples, corresponding to ", longest / 8000, " seconds")

    # Save paths and targets to a CSV file
    data_df = pd.DataFrame({
        'path': paths,
        'valence': targets
    })
    data_df.to_csv(csv_path, index=False)

    print(f"Finished processing and saved csv to {csv_path}")

    return longest_sequence


def spectrograms_with_csv_test(input_dir, output_dir, csv_path="spectrogram.csv", longest_sequence=0):
    """
    input_dir: directory to the raw audio files
    output_dir: directory to save the generated spectrograms
    csv_path: path name of the csv file
    """
    # Load data from input directory
    print(f"Loading audio data from directory: {input_dir}")

    # Prepare to save spectrograms and paths/targets for CSV
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating spectrograms and saving to {output_dir}...")
    paths = []
    targets = []

    # Process each file immediately as it is loaded
    shortest = 1000000
    longest = 0
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith('.pkl'):
            file_path = os.path.join(input_dir, file)
            file_contents = load_pickle(file_path)
            audio_data = file_contents['audio_data']

            # Construct the path for saving the spectrogram
            spectrogram_path = os.path.join(output_dir, f'spectrogram_{len(paths)}.npy')
            audio_to_melspectrogram(audio_data, spectrogram_path, longest_sequence)

            if len(audio_data) < shortest:
                shortest = len(audio_data)
            if len(audio_data) > longest:
                longest = len(audio_data)
            # Append the path and target to lists
            paths.append(spectrogram_path)
            targets.append(file)

    print(f"Padding spectrograms according to the longest sequence: {longest_sequence}...")
    for spectrogram_path in paths:
        pad_spectrograms(spectrogram_path, longest_sequence)

    print("Shortest audio: ", shortest, " samples, corresponding to ", shortest / 8000, " seconds")
    print("Longest audio: ", longest, " samples, corresponding to ", longest / 8000, " seconds")

    # Save paths and targets to a CSV file
    data_df = pd.DataFrame({
        'path': paths,
        'file': targets
    })
    data_df.to_csv(csv_path, index=False)

    print(f"Finished processing and saved csv to {csv_path}")


np.random.seed(config.seed)

# Call spectrograms_with_csv to create the padded spectrograms and csv
input_dir = "../train/"
output_dir = "../mel_spectrograms"
csv_path = "./mel_spectrograms.csv"

longest_sequence = spectrograms_with_csv(input_dir=input_dir,
                     output_dir=output_dir,
                     csv_path=csv_path)


test_input_dir = "../test/"
test_output_dir = "../mel_spectrograms_test"
test_csv_path = "./mel_spectrograms_test.csv"

spectrograms_with_csv_test(input_dir=test_input_dir,
                     output_dir=test_output_dir,
                     csv_path=test_csv_path,
                     longest_sequence=longest_sequence)

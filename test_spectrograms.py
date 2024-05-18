import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('images', exist_ok=True)

# Load your spectrogram data
for i in range(1000, 2000):
    spectrogram = np.load(f'mel_spectrograms/spectrogram_{i}.npy')

    # Number of samples and sampling rate
    total_samples = spectrogram.shape[1] * 512
    sampling_rate = 8000

    # Calculate time per frame
    n_fft = 2048
    hop_length = 512  # or adjust if you used a different value
    time_per_frame = hop_length / sampling_rate
    total_time = total_samples / sampling_rate

    # Generate time values for the x-axis
    frames = spectrogram.shape[1]
    time_ticks = np.linspace(0, total_time, frames)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, total_time, 0, spectrogram.shape[0]])
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f'images/spectrogram_{i}.png')
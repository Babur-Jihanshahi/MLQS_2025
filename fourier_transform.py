import pandas as pd
import numpy as np
from pathlib import Path
from scipy.fft import fft
from scipy.signal import get_window
import matplotlib.pyplot as plt



# We need to apply FFT to the data before downsampling with the time window of 1 second.

SRC = Path("all_modes_1s_mean.csv")  # Source CSV file with resampled data
DST = Path("all_modes_with_patterns.csv")  # Destination CSV file to save the results

def extract_fft_features(df, column, window_size=100, sampling_rate=100):
    """
    Apply FFT on a rolling window and extract dominant frequency and its magnitude.
    
    Args:
        df: DataFrame with time-indexed sensor data
        column: Name of the signal column to use (e.g., 'gyro')
        window_size: Size of the window in seconds (based on 0Hz data, 100 samples for 1s)
        sampling_rate: How many samples per second (Hz)
    
    Returns:
        df with new columns: 'fft_dom_freq', 'fft_dom_magnitude'
    """
    dom_freqs = []
    dom_mags = []

    signal = df[column].fillna(0).to_numpy()
    
    for i in range(len(signal)):
        if i < window_size:
            dom_freqs.append(np.nan)
            dom_mags.append(np.nan)
            continue

        window = signal[i - window_size:i]

        # Optional: apply a windowing function (Hamming, Hann, etc.)
        windowed_signal = window * get_window("hann", window_size)

        # Compute FFT
        freqs = np.fft.fftfreq(window_size, d=1.0 / sampling_rate)
        fft_result = fft(windowed_signal)

        # Consider only the positive frequencies
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_mags = np.abs(fft_result[pos_mask])

        if len(pos_mags) == 0:
            dom_freqs.append(np.nan)
            dom_mags.append(np.nan)
        else:
            dom_idx = np.argmax(pos_mags)
            dom_freqs.append(pos_freqs[dom_idx])
            dom_mags.append(pos_mags[dom_idx])

    df['fft_dom_freq'] = dom_freqs
    df['fft_dom_magnitude'] = dom_mags

    return df

columns_to_use = ['speed','acceleration', 'gyro']  
for col in columns_to_use:
    df = extract_fft_features(df, column=col, window_size=100, sampling_rate=100)

    # Plotting the results to visualize the FFT features
    def plot_fft_features(df, column):
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[column], label=column)
        plt.plot(df.index, df['fft_dom_freq'], label='FFT Dominant Frequency', linestyle='--')
        plt.plot(df.index, df['fft_dom_magnitude'], label='FFT Dominant Magnitude', linestyle=':')
        plt.title(f'FFT Features for {column}')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

# Save final results
df.to_csv(DST)

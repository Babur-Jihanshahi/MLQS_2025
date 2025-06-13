import pandas as pd
import numpy as np
from pathlib import Path
from scipy.fft import fft
from scipy.signal import get_window
import matplotlib.pyplot as plt



def extract_fft_features(df, column, window_size=100, sampling_rate=100):
    """
    Apply FFT on a rolling window and extract dominant frequency and its magnitude.
    
    Args:
        df: DataFrame with time-indexed sensor data
        column: Name of the signal column to use (e.g., 'gyro')
        window_size: Size of the window in seconds (based on 0Hz data, 100 samples for 1s)
        sampling_rate: How many samples per second (Hz)
    
    Returns:
        df with new columns: 'fft_dom_freq_{column}', 'fft_dom_magnitude_{column}'
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

    df[f'fft_dom_freq_{column}'] = dom_freqs
    df[f'fft_dom_magnitude_{column}'] = dom_mags

    return df

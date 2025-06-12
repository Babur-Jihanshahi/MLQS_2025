import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path("all_modes_1s_mean.csv")  # Source CSV file with resampled data
DST = Path("all_modes_with_patterns.csv")  # Destination CSV file to save the results

# Read the CSV file into a DataFrame
df = pd.read_csv(SRC, parse_dates=['timestamp'], index_col='timestamp')

# Patterns for detecting different modes of transportation based on sensor data
PATTERN_FUNCTIONS = {
    # Patterns based on speed and acceleration
    'sustained_low_speed': lambda df: (df['speed'] < 0.5).rolling(window=10).sum() == 10,
    'sustained_medium_speed': lambda df: df['speed'].between(1.5, 3.0).rolling(window=10).sum() == 10,
    'sustained_high_speed': lambda df: (df['speed'] > 3.0).rolling(window=10).sum() == 10,
    # Patterns based on GPS signal
    'consistent_gps_signal': lambda df: df['gps_signal'].notna().rolling(window=10).sum() == 10,
    'gps_signal_loss': lambda df: df['gps_signal'].isna().rolling(window=5).sum() == 5,
    # Patterns based on gyroscope data
    'low_velocity_high_gyro': lambda df: ((df['speed'] < 0.5) & (df['gyro'].abs() > 1.0)).rolling(window=10).sum() == 10,
    'high_velocity_low_gyro': lambda df: ((df['speed'] > 3.0) & (df['gyro'].abs() < 0.5)).rolling(window=10).sum() == 10,
}

def calculate_support(series):
    """Calculate support as the proportion of True values in the Series."""
    return series.sum() / len(series)

# Adding all features to the DataFrame with support threshold
def add_all_features(df, support_threshold=0.05):
    support_log = {}
    for name, func in PATTERN_FUNCTIONS.items():
        series = func(df)
        support = calculate_support(series)
        support_log[name] = support
        if support >= support_threshold:
            df[name] = series
        else:
            print(f"Dropping '{name}' due to low support: {support:.2%}")
    return df, support_log


df, support_log = add_all_features(df)


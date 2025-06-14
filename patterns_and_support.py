import numpy as np
from pathlib import Path
import pandas as pd

SRC = Path("data/final_data.csv")  
DST = Path("data/final_data_with_patterns.csv")  

# Read the CSV file into a DataFrame
df = pd.read_csv(SRC, parse_dates=['timestamp'], index_col='timestamp')

# Gyroscope data processing
df['gyro'] = np.sqrt(df['X (rad/s)']**2 + df['Y (rad/s)']**2 + df['Z (rad/s)']**2)
# Remove original gyroscope columns
df.drop(columns=['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'], inplace=True)

# Dropping GPS data
df.drop(columns=["Latitude (°)","Longitude (°)","Height (m)","Horizontal Accuracy (m)","Vertical Accuracy (°)"], inplace=True)

# Patterns for detecting different modes of transportation based on sensor data
PATTERN_FUNCTIONS = {
    # Patterns based on speed and acceleration
    'sustained_low_speed': lambda df: (df['Velocity (m/s)'] < 0.5).rolling(window=10).sum() == 10,
    'sustained_medium_speed': lambda df: df['Velocity (m/s)'].between(1.5, 3.0).rolling(window=10).sum() == 10,
    'sustained_high_speed': lambda df: (df['Velocity (m/s)'] > 3.0).rolling(window=10).sum() == 10,
    
    # Patterns based on gyroscope data
    'low_velocity_high_gyro': lambda df: ((df['Velocity (m/s)'] < 0.5) & (df['gyro'].abs() > 1.0)).rolling(window=10).sum() == 10,
    'high_velocity_low_gyro': lambda df: ((df['Velocity (m/s)'] > 3.0) & (df['gyro'].abs() < 0.5)).rolling(window=10).sum() == 10,
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
        print(f"Feature '{name}' support: {support:.2%}")
        if support >= support_threshold:
            df[name] = series.fillna(False)
        else:
            print(f"Dropping '{name}' due to low support: {support:.2%}")
    return df, support_log


df, support_log = add_all_features(df)

print("Final DataFrame with patterns and support:")
print(df.head())

# Convert boolean columns to 0/1 for better compatibility
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Save the modified DataFrame to a new CSV file
df.to_csv(DST)
print(f"Data saved to {DST}")


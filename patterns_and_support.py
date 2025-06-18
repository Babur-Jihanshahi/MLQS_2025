import numpy as np
from pathlib import Path
import pandas as pd

SRC = Path("data/final_data.csv")  
DST = Path("data/final_data_with_patterns.csv")  

# Read the CSV file into a DataFrame
df = pd.read_csv(SRC, parse_dates=['timestamp'], index_col='timestamp')

print(f"Original dataset shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

# Gyroscope data processing
df['gyro'] = np.sqrt(df['X (rad/s)']**2 + df['Y (rad/s)']**2 + df['Z (rad/s)']**2)
# Remove original gyroscope columns
df.drop(columns=['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'], inplace=True)

# Drop GPS data AND their FFT features
gps_related_columns = [
    "Latitude (°)", "Longitude (°)", "Height (m)", 
    "Horizontal Accuracy (m)", "Vertical Accuracy (°)",
    # Also remove FFT features of GPS coordinates
    "fft_dom_freq_Latitude (°)", "fft_dom_magnitude_Latitude (°)",
    "fft_dom_freq_Longitude (°)", "fft_dom_magnitude_Longitude (°)",
    "fft_dom_freq_Height (m)", "fft_dom_magnitude_Height (m)"
]

# Only drop columns that actually exist
columns_to_drop = [col for col in gps_related_columns if col in df.columns]
print(f"\nRemoving {len(columns_to_drop)} GPS-related columns:")
for col in columns_to_drop:
    print(f"  - {col}")

df.drop(columns=columns_to_drop, inplace=True)

print(f"Dataset shape after GPS removal: {df.shape}")

# Patterns for detecting different modes of transportation based on sensor data
PATTERN_FUNCTIONS = {
    # Patterns based on speed and acceleration
    'sustained_low_speed': lambda df: (df['Velocity (m/s)'] < 0.5).rolling(window=10).sum() == 10,
    'sustained_medium_speed': lambda df: df['Velocity (m/s)'].between(1.5, 3.0).rolling(window=10).sum() == 10,
    'sustained_high_speed': lambda df: (df['Velocity (m/s)'] > 3.0).rolling(window=10).sum() == 10,
    
    # Patterns based on gyroscope data
    'low_velocity_high_gyro': lambda df: ((df['Velocity (m/s)'] < 0.5) & (df['gyro'].abs() > 1.0)).rolling(window=10).sum() == 10,
    'high_velocity_low_gyro': lambda df: ((df['Velocity (m/s)'] > 3.0) & (df['gyro'].abs() < 0.5)).rolling(window=10).sum() == 10,
    
    # Additional motion patterns
    'acceleration_variance_high': lambda df: df['Y (m/s^2)'].rolling(window=10).std() > 2.0,
    'acceleration_variance_low': lambda df: df['Y (m/s^2)'].rolling(window=10).std() < 0.5,
}

def calculate_support(series):
    """Calculate support as the proportion of True values in the Series."""
    return series.sum() / len(series)

# Adding all features to the DataFrame with support threshold
def add_all_features(df, support_threshold=0.05):
    support_log = {}
    for name, func in PATTERN_FUNCTIONS.items():
        try:
            series = func(df)
            support = calculate_support(series)
            support_log[name] = support
            print(f"Feature '{name}' support: {support:.2%}")
            if support >= support_threshold:
                df[name] = series.fillna(False)
            else:
                print(f"Dropping '{name}' due to low support: {support:.2%}")
        except Exception as e:
            print(f"Error creating feature '{name}': {e}")
    return df, support_log

print(f"\nCreating motion pattern features...")
df, support_log = add_all_features(df)

print(f"\nFinal DataFrame with patterns:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Convert boolean columns to 0/1 for better compatibility
bool_cols = df.select_dtypes(include='bool').columns
if len(bool_cols) > 0:
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"Converted {len(bool_cols)} boolean columns to integers")

# Check remaining features
print(f"\nRemaining feature categories:")
sensor_features = [col for col in df.columns if any(x in col for x in ['(m/s^2)', 'gyro'])]
motion_features = [col for col in df.columns if any(x in col for x in ['Velocity', 'Direction'])]
fft_features = [col for col in df.columns if 'fft_dom' in col]
pattern_features = [col for col in df.columns if any(x in col for x in ['sustained', 'velocity', 'acceleration'])]

print(f"  - Sensor features: {len(sensor_features)} - {sensor_features}")
print(f"  - Motion features: {len(motion_features)} - {motion_features}")
print(f"  - FFT features: {len(fft_features)} - {fft_features[:5]}{'...' if len(fft_features) > 5 else ''}")
print(f"  - Pattern features: {len(pattern_features)} - {pattern_features}")

# Save the modified DataFrame to a new CSV file
df.to_csv(DST)
print(f"\nData saved to {DST}")

# Verification check
print(f"\nVerification - GPS coordinates removed:")
gps_check = ['Latitude', 'Longitude', 'Height']
for coord in gps_check:
    remaining = [col for col in df.columns if coord in col]
    if remaining:
        print(f"  ⚠️  Still found {coord}-related columns: {remaining}")
    else:
        print(f"  ✅ No {coord}-related columns found")
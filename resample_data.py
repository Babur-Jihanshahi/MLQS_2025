from pathlib import Path
import pandas as pd
import numpy as np


SRC = Path("all_modes_raw_long.csv")      
DST = Path("all_modes_1s_mean.csv")
DT  = "1S"                                
ANCHOR = pd.Timestamp("2025-01-01 00:00")  

print(f"reading {SRC} …")
df = pd.read_csv(SRC)

# locate the time index column
time_candidates = [
    c for c in df.columns
    if c in {"time", "t"} or c.endswith(("_time", "_t"))
]
if not time_candidates:
    raise ValueError("No time-index column found "
                     "(expected 'time', 't', '*_time' or '*_t').")

# Prefer plain 'time' if present, otherwise take the first candidate
t_col = "time" if "time" in time_candidates else time_candidates[0]
print(f"using '{t_col}' as the time column")

# build timestamp index & sort 
df["timestamp"] = ANCHOR + pd.to_timedelta(df[t_col], unit="s")
df = df.set_index("timestamp").sort_index()

# resample numeric columns to 60-s mean 
num_cols = df.select_dtypes("number").columns
resampled = df[num_cols].resample(DT).mean()

# save
resampled.to_csv(DST, index_label="timestamp")
print(f"Wrote {DST}  ({resampled.shape[0]} rows × {resampled.shape[1]} cols)")

import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path("gps_gyro_data")           

TRANSPORT_FILES = {
    "walking": [
        "Walking_GPS_2025_06_04_19_35_39.csv",
        "Walking_gyro.csv",
    ],
    "running": [
        "Runing_GPS_2025_06_04_9_02_29AM.csv",
        "Running_GPS_2025_06_03_9_18_12AM.csv",
    ],
    "train" : ["Train_GPS_2025_06_02_14_45_38.csv"],
    "biking": ["Biking_GPS_2025_06_02_2_44_17PM.csv"],
}

def tidy(df):
    df = df.copy()
    df.columns = (df.columns.str.lower().str.strip()
                  .str.replace(r"\(.*?\)", "", regex=True)
                  .str.replace("[^a-z0-9]+", "_", regex=True)
                  .str.strip("_"))
    return df

def gyro_has_header(path: Path):
    try:
        [float(c) for c in pd.read_csv(path, nrows=1).columns]
        return False
    except ValueError:
        return True

records = []
for mode, files in TRANSPORT_FILES.items():
    for fname in files:
        fp = DATA_DIR / fname
        if not fp.exists():
            print("missing:", fp); continue

        is_gyro = "gyro" in fname.lower()
        if is_gyro:
            if gyro_has_header(fp):
                df = pd.read_csv(fp)
            else:
                n = len(pd.read_csv(fp, nrows=1, header=None).columns)
                cols = ["t", "x", "y", "z"] if n == 4 else ["t", "x", "y", "z", "mag"]
                df = pd.read_csv(fp, header=None, names=cols)
        else:
            df = pd.read_csv(fp)

        df = tidy(df)
        if is_gyro:
            df = df.add_prefix("gyro_")

        df["mode"]   = mode
        df["sensor"] = "gyro" if is_gyro else "gps"
        records.append(df)

all_data = pd.concat(records, ignore_index=True, sort=False)
all_data.to_csv("all_modes_raw_long.csv", index=False)
print("saved → all_modes_raw_long.csv  (#rows =", len(all_data), ")")

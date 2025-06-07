import re
from pathlib import Path
import numpy as np
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
    "train" : [ "Train_GPS_2025_06_02_14_45_38.csv" ],
    "biking": [ "Biking_GPS_2025_06_02_2_44_17PM.csv" ],
}


def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.lower().str.strip()
                  .str.replace(r"\(.*?\)", "", regex=True)
                  .str.replace("[^a-z0-9]+", "_", regex=True)
                  .str.strip("_"))
    return df

def gyro_has_header(path: Path) -> bool:
    try:
        [float(c) for c in pd.read_csv(path, nrows=1).columns]
        return False   # headerless
    except ValueError:
        return True    # header present


records = []

for mode, files in TRANSPORT_FILES.items():
    for fname in files:
        fp = DATA_DIR / fname
        if not fp.exists():
            print(f"Missing: {fp}")
            continue

        is_gyro = "gyro" in fname.lower()

        if is_gyro:
            if gyro_has_header(fp):
                df = pd.read_csv(fp)
            else:
                n_cols = len(pd.read_csv(fp, nrows=1, header=None).columns)
                col_names = ["t", "x", "y", "z"] if n_cols == 4 else \
                            ["t", "x", "y", "z", "mag"]
                df = pd.read_csv(fp, header=None, names=col_names)
        else:
            df = pd.read_csv(fp)

        df = tidy_columns(df)

        # prefix every gyro column with 'gyro_' to avoid clashes
        if is_gyro:
            df = df.add_prefix("gyro_")

        df["mode"]   = mode
        records.append(df)

all_data = pd.concat(records, ignore_index=True, sort=False)
print("Rows loaded:", all_data.shape)


summary_rows = []

for mode in all_data["mode"].unique():
    sub = all_data[all_data["mode"] == mode]
    for col in sub.select_dtypes(include=[np.number]).columns:
        s = sub[col]
        summary_rows.append({
            "mode"        : mode,
            "feature"     : col,
            "missing_pct" : s.isna().mean() * 100,
            "mean"        : s.mean(),
            "median"      : s.median(),
            "std"         : s.std(),
            "min"         : s.min(),
            "max"         : s.max(),
        })

summary = (pd.DataFrame(summary_rows)
             .sort_values(["mode", "feature"])
             .reset_index(drop=True))

summary.to_csv("mode_numeric_summary.csv", index=False)
print("mode_numeric_summary.csv written")

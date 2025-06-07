import pandas as pd
summary = pd.read_csv("transport_mode_features.csv", index_col="mode")
print(summary.head())

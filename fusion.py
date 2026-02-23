"""
Data fusion: merge internet_data + realtime_data for PatchTST.

- internet_data.csv = built by build_internet_data.py (BMKG + lettuce: weather/temp + soil moisture/pH).
- realtime_data.csv = dummy data when no live sensors; used as last input window for prediction.

Training: load_and_merge() → merged series → preprocess → train PatchTST.
Prediction: realtime_data (last INPUT_LEN rows) → PatchTST → outcome.
"""
import pandas as pd

def load_and_merge():
    internet = pd.read_csv("data/internet_data.csv")
    realtime = pd.read_csv("data/realtime_data.csv")
    merged = pd.concat([internet, realtime], ignore_index=True)
    return merged

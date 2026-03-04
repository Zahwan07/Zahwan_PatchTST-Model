"""
Predict using the Main_model model trained on hourly data.
Cuaca is one-hot encoded; output is inverse-transformed to [suhu, cuaca, kelembapan, ph].

Run from project root:  python Main_model/predict_exp.py

Uses last 168 hours from data/historical_environment.csv as input.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import pandas as pd
import joblib

from model.patchtst_official import PatchTST_Official
from Main_model.config_exp import FEATURES_FORECAST, PRED_LEN, INPUT_LEN, ALL_FEATURES

# Paths
OUT_DIR = os.path.join(ROOT, "Main_model")
DATA_PATH = os.path.join(ROOT, "data", "historical_environment.csv")
MODEL_PATH = os.path.join(OUT_DIR, "patchtst_model_exp.pth")
PREPROCESSOR_PATH = os.path.join(OUT_DIR, "preprocessor_exp.joblib")

# Load preprocessor and model
preprocessor = joblib.load(PREPROCESSOR_PATH)
INPUT_DIM = preprocessor.n_features_in_

model = PatchTST_Official(
    input_dim=INPUT_DIM,
    input_len=INPUT_LEN,
    pred_len=PRED_LEN,
    patch_len=16,
    stride=8,
    d_model=128,
    n_heads=16,
    num_layers=3,
    dropout=0.2,
    revin=True,
)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Load data, take last 168 hours, transform (cuaca -> one-hot)
df = pd.read_csv(DATA_PATH)
input_window_df = df.tail(INPUT_LEN)
input_scaled = preprocessor.transform(input_window_df)

input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    pred_scaled = model(input_tensor)

# Inverse transform -> full feature set
pred_np = pred_scaled.numpy().reshape(-1, INPUT_DIM)
pred_full = preprocessor.inverse_transform(pred_np).reshape(PRED_LEN, -1)

# Build timestamps (first prediction = 1 hour after last input row)
last_time = pd.to_datetime(df["time"].iloc[-1])
pred_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=PRED_LEN, freq="h")

# Save forecast CSV with main columns [suhu, cuaca, kelembapan, ph]
col_idx = [ALL_FEATURES.index(c) for c in FEATURES_FORECAST if c in ALL_FEATURES]
pred_raw = pred_full[:, col_idx] if col_idx else pred_full[:, :4]
pred_df = pd.DataFrame(
    pred_raw,
    columns=FEATURES_FORECAST,
    index=pred_times,
)
pred_df.index.name = "time"
pred_path = os.path.join(OUT_DIR, "prediction_forecast.csv")
pred_df.to_csv(pred_path)
print(f"Full forecast saved to: {pred_path}")

# Console: clearly labeled PREDICTED values
print("\n" + "=" * 60)
print("PREDICTED FORECAST (model output — next hours ahead)")
print("=" * 60)
print(f"Based on: last {INPUT_LEN} hours of data (ending {last_time})")
print(f"Predicted: next {PRED_LEN} hours")
print("\n--- PREDICTED values (sample every 24h) ---")
for d in range(min(7, (PRED_LEN + 23) // 24)):
    h = d * 24
    if h < PRED_LEN:
        row = pred_raw[h]
        ts = pred_times[h].strftime("%Y-%m-%d %H:%M")
        cuaca_name = {0: "clear", 1: "cloudy", 2: "rain"}.get(int(row[1]), int(row[1]))
        print(f"  {ts} | suhu={row[0]:.1f}°C, cuaca={cuaca_name}, kelembapan={row[2]:.1%}, ph={row[3]:.2f}")
print("\n(kelembapan 0.36 = 36% moisture; cuaca: 0=clear, 1=cloudy, 2=rain)")
print("See Main_model/HOW_TO_READ_FORECAST.md for full interpretation guide.")

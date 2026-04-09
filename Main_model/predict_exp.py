"""
Predict using the Main_model model trained on hourly data.
Output: [suhu, cuaca, humidity, light_intensity, ph, precipitation]; cuaca is derived from predicted precipitation.

Run from project root:  python Main_model/predict_exp.py
Uses last 168 hours from data/historical_environment.csv as input.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import pandas as pd
import joblib

from model.patchtst_official import PatchTST_Official
from Main_model.config_exp import (
    FEATURES_FORECAST,
    PRED_LEN,
    INPUT_LEN,
    ALL_FEATURES,
)
# from Main_model.suhu_bias_calib import load_suhu_hourly_bias_c

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

# Load data, take last 168 hours, transform (all continuous; cuaca not in model)
df = pd.read_csv(DATA_PATH)
input_window_df = df.tail(INPUT_LEN)
input_scaled = preprocessor.transform(input_window_df)

input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    pred_scaled = model(input_tensor)

# Inverse transform -> full feature set
pred_np = pred_scaled.numpy().reshape(-1, INPUT_DIM)
pred_full = preprocessor.inverse_transform(pred_np).reshape(PRED_LEN, -1)

# Derive cuaca from predicted precipitation (0=clear, 1=cloudy, 2=rain)
idx = {c: ALL_FEATURES.index(c) for c in ("suhu", "humidity", "light_intensity", "ph", "precipitation")}
pred_precipitation = pred_full[:, idx["precipitation"]]
cuaca_derived = np.where(pred_precipitation > 0.5, 2, np.where(pred_precipitation > 0.1, 1, 0))

# Build timestamps (first prediction = 1 hour after last row in historical_environment.csv)
last_time = pd.to_datetime(df["time"].iloc[-1])
pred_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=PRED_LEN, freq="h")
print(
    f"History ends: {last_time}  |  Forecast range: {pred_times[0]}  →  {pred_times[-1]}  ({PRED_LEN} h)"
)

# Forecast CSV columns: cuaca is not in ALL_FEATURES — assemble explicitly
pred_suhu = pred_full[:, idx["suhu"]].copy()
pred_light = pred_full[:, idx["light_intensity"]]
hours_arr = pred_times.hour.to_numpy()
# Hourly bias disabled per adviser request (keep commented for easy re-enable later).
# bias_c = load_suhu_hourly_bias_c(OUT_DIR)
# if bias_c is not None:
#     pred_suhu = pred_suhu + bias_c[hours_arr]

pred_raw = np.column_stack(
    [
        pred_suhu,
        cuaca_derived.astype(float),
        pred_full[:, idx["humidity"]],
        pred_light,
        pred_full[:, idx["ph"]],
        pred_full[:, idx["precipitation"]],
    ]
)
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
n_days = (PRED_LEN + 23) // 24
print(f"\n--- PREDICTED values (sample every 24h, {n_days} days total) ---")
for d in range(min(5, n_days)):
    h = d * 24
    if h < PRED_LEN:
        row = pred_raw[h]
        ts = pred_times[h].strftime("%Y-%m-%d %H:%M")
        cuaca_name = {0: "clear", 1: "cloudy", 2: "rain"}.get(int(row[1]), int(row[1]))
        print(f"  {ts} | suhu={row[0]:.1f}°C, cuaca={cuaca_name}, humidity={row[2]:.1%}, light={row[3]:.0f} W/m², ph={row[4]:.2f}, precip={row[5]:.2f} mm")
if n_days > 7:
    print("  ...")
    for d in range(max(n_days - 3, 5), n_days):
        h = d * 24
        if h < PRED_LEN:
            row = pred_raw[h]
            ts = pred_times[h].strftime("%Y-%m-%d %H:%M")
            cuaca_name = {0: "clear", 1: "cloudy", 2: "rain"}.get(int(row[1]), int(row[1]))
            print(f"  {ts} | suhu={row[0]:.1f}°C, cuaca={cuaca_name}, humidity={row[2]:.1%}, light={row[3]:.0f} W/m², ph={row[4]:.2f}, precip={row[5]:.2f} mm")
print("\n(cuaca derived from precipitation: >0.5 mm=rain, >0.1 mm=cloudy, else clear; precip=mm/h)")
print("(suhu: raw model output, no hourly-bias correction)")
print("See Main_model/HOW_TO_READ_FORECAST.md for interpretation.")

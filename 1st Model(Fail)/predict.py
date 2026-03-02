"""
Run PatchTST on the last INPUT_LEN rows of realtime_data to get a 7-day
environment forecast, then send the numeric forecast to an LLM for
natural-language recommendations.
"""
import torch
import pandas as pd
import joblib
from model.patchtst import PatchTST
from config import FEATURES, INPUT_DIM, PRED_LEN, INPUT_LEN
from llm_recommendations import forecast_to_text, get_recommendations_from_forecast

# Load model and scaler
model = PatchTST(input_dim=INPUT_DIM, pred_len=PRED_LEN)
model.load_state_dict(torch.load("patchtst_model.pth"))
model.eval()
scaler = joblib.load("scaler.joblib")

# Load realtime data (dummy or live)
df = pd.read_csv("data/realtime_data.csv")
data = df[FEATURES].values
data_scaled = scaler.transform(data)
input_window = data_scaled[-INPUT_LEN:]

# Predict next 7 days × 4 variables
input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    pred_scaled = model(input_tensor)

# Inverse transform to original scale (pred_scaled: 1, 7, 4)
pred_np = pred_scaled.numpy().reshape(-1, 4)
pred_raw = scaler.inverse_transform(pred_np).reshape(1, PRED_LEN, INPUT_DIM)[0]

# Build 7-day forecast list of dicts for LLM
forecast_7d = [
    {"suhu": float(pred_raw[d, 0]), "cuaca": float(pred_raw[d, 1]), "kelembapan": float(pred_raw[d, 2]), "ph": float(pred_raw[d, 3])}
    for d in range(PRED_LEN)
]

# Print numeric forecast
print("=" * 60)
print("7-DAY ENVIRONMENT FORECAST (red leaf lettuce, Bandung Coblong)")
print("=" * 60)
print(forecast_to_text(forecast_7d))
print()

# LLM recommendations
print("=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
recommendations = get_recommendations_from_forecast(
    forecast_7d,
    crop="red leaf lettuce (Lactuca sativa L.)",
    location="Bandung (Coblong)",
)
print(recommendations)

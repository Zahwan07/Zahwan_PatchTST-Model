"""
Localhost server: current conditions + PatchTST forecast + LLM recommendations.
Uses actual time: reads prediction_forecast.csv and shows the row for the current hour.
Refreshes every hour so the recommendation and current-hour data update on the hour.

Run from project root:  python serve_realtime.py
Then open http://127.0.0.1:5000 in your browser.
"""
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Default to local Ollama (qwen2.5:14b) so recommendations work when server is started from IDE or without env vars
os.environ.setdefault("LLM_API_BASE", "http://localhost:11434/v1")
os.environ.setdefault("LLM_MODEL", "qwen2.5:14b")

import numpy as np
import torch
import pandas as pd
import joblib
from flask import Flask

from config import (
    LETTUCE_TEMP_OPTIMAL,
    LETTUCE_HUMIDITY_OPTIMAL,
    LETTUCE_LIGHT_OPTIMAL,
    LETTUCE_PH_OPTIMAL,
)
from model.patchtst_official import PatchTST_Official
from Main_model.config_exp import FEATURES_FORECAST, PRED_LEN, INPUT_LEN, ALL_FEATURES
# from Main_model.suhu_bias_calib import load_suhu_hourly_bias_c
from llm_recommendations import get_recommendations_from_current_conditions

app = Flask(__name__)
REALTIME_PATH = os.path.join(ROOT, "data", "realtime_data.csv")
HISTORICAL_PATH = os.path.join(ROOT, "data", "historical_environment.csv")
OUT_DIR = os.path.join(ROOT, "Main_model")
PREDICTION_FORECAST_PATH = os.path.join(OUT_DIR, "prediction_forecast.csv")
MODEL_PATH = os.path.join(OUT_DIR, "patchtst_model_exp.pth")
PREPROCESSOR_PATH = os.path.join(OUT_DIR, "preprocessor_exp.joblib")

# Refresh every hour so recommendation and "current hour" data update on the hour
REFRESH_SECONDS = 3600

# ---- Load PatchTST model and preprocessor once at startup ----
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
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

CUACA_NAMES = {0: "clear", 1: "cloudy", 2: "rain"}

# Optimal ranges for "is optimal" check
T_LO, T_HI = LETTUCE_TEMP_OPTIMAL
H_LO, H_HI = LETTUCE_HUMIDITY_OPTIMAL
L_LO, L_HI = LETTUCE_LIGHT_OPTIMAL
P_LO, P_HI = LETTUCE_PH_OPTIMAL


def is_optimal(suhu, humidity, light_intensity, ph):
    """True if all values are within optimal ranges for red leaf lettuce (hydroponic)."""
    return (
        T_LO <= suhu <= T_HI
        and H_LO <= humidity <= H_HI
        and L_LO <= light_intensity <= L_HI
        and P_LO <= ph <= P_HI
    )


def get_current_hour_realtime():
    """
    Get the row from realtime_data.csv that matches the current hour (local time).
    So "realtime" updates every hour by which row we use. Returns (suhu, cuaca, humidity, light_intensity, ph) or None.
    """
    if not os.path.isfile(REALTIME_PATH):
        return None
    df = pd.read_csv(REALTIME_PATH)
    if "time" not in df.columns:
        row = df.iloc[-1]
        return (
            float(row["suhu"]),
            int(row["cuaca"]),
            float(row.get("humidity", row.get("kelembapan", 0))),
            float(row.get("light_intensity", 0)),
            float(row["ph"]),
        )
    now = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    df["time"] = pd.to_datetime(df["time"])
    match = df[
        (df["time"].dt.year == current_hour.year)
        & (df["time"].dt.month == current_hour.month)
        & (df["time"].dt.day == current_hour.day)
        & (df["time"].dt.hour == current_hour.hour)
    ]
    if match.empty:
        df = df.sort_values("time")
        idx = (df["time"] - pd.Timestamp(current_hour)).abs().idxmin()
        row = df.loc[idx]
    else:
        row = match.iloc[0]
    return (
        float(row["suhu"]),
        int(row["cuaca"]),
        float(row.get("humidity", row.get("kelembapan", 0))),
        float(row.get("light_intensity", 0)),
        float(row["ph"]),
    )


def count_optimal_hours_ahead():
    """
    From prediction_forecast.csv, count how many consecutive hours starting from current hour are optimal.
    Returns (count, max_hours_checked). Uses Main_model/prediction_forecast.csv.
    """
    if not os.path.isfile(PREDICTION_FORECAST_PATH):
        return 0, 0
    df = pd.read_csv(PREDICTION_FORECAST_PATH)
    if "time" not in df.columns:
        return 0, 0
    now = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    df["time"] = pd.to_datetime(df["time"])
    # Find index of current hour
    match = df[
        (df["time"].dt.year == current_hour.year)
        & (df["time"].dt.month == current_hour.month)
        & (df["time"].dt.day == current_hour.day)
        & (df["time"].dt.hour == current_hour.hour)
    ]
    if match.empty:
        return 0, 0
    start_idx = match.index[0]
    count = 0
    max_check = min(48, len(df) - start_idx)  # check up to 48 hours ahead
    for i in range(max_check):
        row = df.iloc[start_idx + i]
        s = float(row["suhu"])
        h = float(row.get("humidity", row.get("kelembapan", 0)))
        li = float(row.get("light_intensity", 0))
        p = float(row["ph"])
        if is_optimal(s, h, li, p):
            count += 1
        else:
            break
    return count, max_check


def get_current_hour_prediction():
    """
    Get the row from prediction_forecast.csv that matches the current hour (local time).
    Returns (time_str, suhu, cuaca_str, humidity, light_intensity, ph) or (None, ...) if not found.
    """
    now = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    if not os.path.isfile(PREDICTION_FORECAST_PATH):
        return None, None, None, None, None, None
    df = pd.read_csv(PREDICTION_FORECAST_PATH)
    df["time"] = pd.to_datetime(df["time"])
    match = df[
        (df["time"].dt.year == current_hour.year)
        & (df["time"].dt.month == current_hour.month)
        & (df["time"].dt.day == current_hour.day)
        & (df["time"].dt.hour == current_hour.hour)
    ]
    if match.empty:
        df = df.sort_values("time")
        idx = (df["time"] - pd.Timestamp(current_hour)).abs().idxmin()
        row = df.loc[idx]
    else:
        row = match.iloc[0]
    time_str = row["time"].strftime("%Y-%m-%d %H:%M") if hasattr(row["time"], "strftime") else str(row["time"])
    suhu = float(row["suhu"])
    cuaca = int(round(row["cuaca"])) if pd.notna(row["cuaca"]) else 0
    humidity = float(row.get("humidity", row.get("kelembapan", 0)))
    light_intensity = float(row.get("light_intensity", 0))
    ph = float(row["ph"])
    return time_str, suhu, CUACA_NAMES.get(cuaca, str(cuaca)), humidity, light_intensity, ph


def get_patchtst_forecast():
    """Run PatchTST on last 168h of historical_environment; return short forecast summary (next 24h, next 7d)."""
    df = pd.read_csv(HISTORICAL_PATH)
    input_window_df = df.tail(INPUT_LEN)
    input_scaled = preprocessor.transform(input_window_df)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(input_tensor)
    pred_np = pred_scaled.numpy().reshape(-1, INPUT_DIM)
    pred_full = preprocessor.inverse_transform(pred_np).reshape(PRED_LEN, -1)
    idx = {c: ALL_FEATURES.index(c) for c in ("suhu", "humidity", "light_intensity", "ph", "precipitation")}
    pred_precipitation = pred_full[:, idx["precipitation"]]
    cuaca_derived = np.where(pred_precipitation > 0.5, 2, np.where(pred_precipitation > 0.1, 1, 0))
    pred_light = pred_full[:, idx["light_intensity"]]
    last_time = pd.to_datetime(df["time"].iloc[-1])
    pred_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=PRED_LEN,
        freq="h",
    )
    hours_arr = pred_times.hour.to_numpy()
    pred_suhu = pred_full[:, idx["suhu"]].copy()
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
    cuaca_names = {0: "clear", 1: "cloudy", 2: "rain"}
    # Next 24h: sample at hour 0 and 12 (suhu, cuaca, humidity, light_intensity, ph)
    lines = []
    lines.append("Next 24h (model): " + ", ".join([
        f"{pred_raw[0, 0]:.1f}°C {cuaca_names.get(int(pred_raw[0, 1]), int(pred_raw[0, 1]))} humidity {pred_raw[0, 2]:.2f} light {pred_raw[0, 3]:.0f} W/m² pH {pred_raw[0, 4]:.2f}",
        f"12h: {pred_raw[12, 0]:.1f}°C humidity {pred_raw[12, 2]:.2f} light {pred_raw[12, 3]:.0f}",
    ]))
    # Next 7 days: sample one per day at hour 0
    day7 = [f"day{i+1} {pred_raw[i*24, 0]:.1f}°C" for i in range(min(7, PRED_LEN // 24))]
    lines.append("Next 7 days (model): " + ", ".join(day7))
    return "\n".join(lines)


def get_current_data():
    """Read realtime for current hour + forecast, optimal check, get Qwen recommendation."""
    # Current conditions: use row for current hour from realtime_data (hourly-updated by which row we pick)
    rt = get_current_hour_realtime()
    if rt is None:
        try:
            df_rt = pd.read_csv(REALTIME_PATH)
            row = df_rt.iloc[-1]
            suhu = float(row["suhu"])
            humidity = float(row.get("humidity", row.get("kelembapan", 0)))
            light_intensity = float(row.get("light_intensity", 0))
            ph = float(row["ph"])
            cuaca = int(row["cuaca"])
        except Exception:
            suhu, humidity, light_intensity, ph, cuaca = 22.0, 0.55, 300.0, 6.5, 0
    else:
        suhu, cuaca, humidity, light_intensity, ph = rt
    cuaca_str = {0: "clear", 1: "cloudy", 2: "rain"}.get(cuaca, str(cuaca))

    t_lo, t_hi = LETTUCE_TEMP_OPTIMAL
    h_lo, h_hi = LETTUCE_HUMIDITY_OPTIMAL
    l_lo, l_hi = LETTUCE_LIGHT_OPTIMAL
    p_lo, p_hi = LETTUCE_PH_OPTIMAL

    current_optimal = is_optimal(suhu, humidity, light_intensity, ph)
    optimal_hours, _ = count_optimal_hours_ahead()

    today_summary = (
        f"Today's temperature is {suhu:.1f}°C. Weather: {cuaca_str}. "
        f"Humidity {humidity:.2f} (0–1). Light intensity {light_intensity:.0f} W/m². pH is {ph:.2f}. "
        f"Optimal for red leaf lettuce (hydroponic): temperature {t_lo}–{t_hi}°C, "
        f"humidity {h_lo:.2f}–{h_hi:.2f}, light {l_lo}–{l_hi} W/m², pH {p_lo}–{p_hi}. "
        f"Current conditions optimal: {'Yes' if current_optimal else 'No'}. "
        f"Predicted conditions remain optimal for the next {optimal_hours} hours."
    )

    # PatchTST forecast summary (next 24h, next 7d)
    try:
        forecast_summary = get_patchtst_forecast()
    except Exception as e:
        forecast_summary = f"(Model forecast unavailable: {e})"

    # Current hour row from prediction_forecast.csv (for display)
    try:
        current_hour_row = get_current_hour_prediction()
    except Exception as e:
        current_hour_row = (None, None, None, None, None, None)

    # If current and next several hours are optimal, ask LLM to say "optimal shape, check again in X hours"
    full_summary = today_summary + " " + forecast_summary
    recommendation = get_recommendations_from_current_conditions(
        full_summary, optimal_hours=optimal_hours, current_optimal=current_optimal
    )
    return today_summary, forecast_summary, current_hour_row, recommendation


def _esc(s):
    if not s:
        return ""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


@app.route("/")
def index():
    now = datetime.now().strftime("%Y-%m-%d %H:%M (local)")
    try:
        today_summary, forecast_summary, current_hour_row, recommendation = get_current_data()
    except Exception as e:
        today_summary = ""
        forecast_summary = f"Error: {e}"
        current_hour_row = (None, None, None, None, None, None)
        recommendation = f"Could not load recommendation: {e}"
        import traceback
        traceback.print_exc()

    time_str, pred_suhu, pred_cuaca, pred_humidity, pred_light, pred_ph = current_hour_row
    if time_str is not None and pred_suhu is not None:
        current_hour_html = (
            f"<strong>Predicted for this hour</strong> ({time_str}, from prediction_forecast.csv)<br>"
            f"Temperature {pred_suhu:.1f}°C, {pred_cuaca}, humidity {pred_humidity:.2f} (0–1), light {pred_light:.0f} W/m², pH {pred_ph:.2f}."
        )
    else:
        current_hour_html = "No prediction row for the current hour in prediction_forecast.csv (run predict_exp.py and ensure forecast covers today)."

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="{REFRESH_SECONDS}">
  <title>ModelTA — PatchTST + Realtime + LLM</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ font-size: 1.25rem; color: #333; }}
    .summary, .forecast, .current-hour {{ background: #f5f5f5; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
    .forecast {{ background: #e8f4f8; }}
    .current-hour {{ background: #e8f8e8; }}
    .recommendation {{ white-space: pre-wrap; line-height: 1.5; }}
    .meta {{ color: #666; font-size: 0.875rem; margin-top: 1.5rem; }}
  </style>
</head>
<body>
  <h1>Red leaf lettuce — PatchTST forecast, current conditions & recommendations</h1>
  <p class="meta">Last updated: {now} · page refreshes every hour</p>
  <section class="current-hour">
    {current_hour_html}
  </section>
  <section class="summary">
    <strong>Current conditions (realtime)</strong><br>
    {_esc(today_summary)}
  </section>
  <section class="forecast">
    <strong>Model forecast (PatchTST)</strong><br>
    {_esc(forecast_summary)}
  </section>
  <section class="recommendation">
    <strong>Recommendation</strong><br>
    {_esc(recommendation)}
  </section>
</body>
</html>"""
    return html


if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    print("Uses patchtst_model_exp.pth. Page refreshes every hour. Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5000, debug=False)

"""
Sync data/realtime_data.csv from a prediction forecast CSV (e.g. 2-week).
Run from project root:  python data/sync_realtime_from_forecast.py

Uses prediction_forecast 2 week.csv by default; set env REALTIME_SOURCE to another path.
Output: data/realtime_data.csv with columns time, suhu, cuaca, humidity, light_intensity, ph.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(ROOT, "prediction_forecast 2 week.csv")
OUT_PATH = os.path.join(ROOT, "data", "realtime_data.csv")


def main():
    path = os.environ.get("REALTIME_SOURCE", DEFAULT_SOURCE)
    if not os.path.isfile(path):
        print(f"Source not found: {path}")
        return
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df["humidity"] = df["humidity"] if "humidity" in df.columns else df["kelembapan"]
    if "light_intensity" not in df.columns:
        hour = df["time"].dt.hour
        df["light_intensity"] = 0.0
        mask = (hour >= 6) & (hour <= 18)
        df.loc[mask, "light_intensity"] = 150 + (hour[mask] - 6) / 12 * 350
    df = df[["time", "suhu", "cuaca", "humidity", "light_intensity", "ph"]]
    df.to_csv(OUT_PATH, index=False)
    print(f"Written {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()

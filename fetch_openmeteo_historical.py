"""
Fetch hourly historical weather from Open-Meteo (Bandung).
Output: data/historical_environment.csv for hydroponic red leaf lettuce.

Features:
- From API: temperature_2m, relative_humidity_2m, dewpoint_2m, surface_pressure,
  cloud_cover, shortwave_radiation, wind_speed_10m, wind_direction_10m, precipitation,
  weather_code
- Derived: cuaca (weather), humidity (air, 0–1), light_intensity (shortwave W/m²), ph,
  wind_dir_sin/cos, hour_sin/cos, doy_sin/cos, temp_lag_24/168.

No soil moisture (hydroponics). Main prediction outputs: Temperature, Humidity, Weather, Light intensity, pH.
"""
import requests
import pandas as pd
import numpy as np

# Bandung coordinates
LAT = -6.8783
LON = 107.6219

OUTPUT_PATH = "data/historical_environment.csv"

# Fetch in yearly chunks to avoid API limits
YEAR_CHUNKS = [
    ("2020-01-01", "2020-12-31"),
    ("2021-01-01", "2021-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
    ("2026-01-01", "2026-05-31"),
]

# Open-Meteo hourly parameters for better temperature/weather context
HOURLY_PARAMS = (
    "temperature_2m,relative_humidity_2m,dewpoint_2m,"
    "surface_pressure,cloud_cover,shortwave_radiation,"
    "wind_speed_10m,wind_direction_10m,precipitation,weather_code"
)


def fetch_openmeteo_hourly(start: str, end: str) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={start}&end_date={end}"
        f"&hourly={HOURLY_PARAMS}"
        "&timezone=Asia%2FJakarta"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    data = response.json()["hourly"]
    return pd.DataFrame(data)


def fetch_openmeteo():
    """Fetch hourly data (chunked by year)."""
    dfs = []
    for start, end in YEAR_CHUNKS:
        print(f"Fetching {start} to {end}...")
        df = fetch_openmeteo_hourly(start, end)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_sin, hour_cos, doy_sin, doy_cos for time encoding."""
    times = pd.to_datetime(df["time"])
    hour = times.dt.hour
    doy = times.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def add_wind_direction_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind_dir_sin, wind_dir_cos from wind_direction_10m (degrees)."""
    wd = df["wind_direction_10m"].fillna(0).values
    wd_rad = np.deg2rad(wd)
    df["wind_dir_sin"] = np.sin(wd_rad)
    df["wind_dir_cos"] = np.cos(wd_rad)
    return df


def add_temp_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add temp_lag_24 and temp_lag_168 (temperature 24h and 168h ago)."""
    temp = df["temperature_2m"].values
    df["temp_lag_24"] = np.roll(temp, 24)
    df["temp_lag_168"] = np.roll(temp, 168)
    # First rows have rollover from end; mask with NaN then forward-fill from row 168
    df.loc[:23, "temp_lag_24"] = np.nan
    df.loc[:167, "temp_lag_168"] = np.nan
    df["temp_lag_24"] = df["temp_lag_24"].bfill().fillna(temp[0])
    df["temp_lag_168"] = df["temp_lag_168"].bfill().fillna(temp[0])
    return df


def generate_ph(df: pd.DataFrame) -> pd.DataFrame:
    """Slow drift in pH (6.0–7.0)."""
    np.random.seed(123)
    ph = []
    base = 6.5
    for _ in range(len(df)):
        drift = np.random.normal(0, 0.01)
        base = base + drift
        base = np.clip(base, 6.0, 7.0)
        ph.append(base)
    df["ph"] = ph
    return df


def build_dataset():
    df = fetch_openmeteo()

    # Cuaca: 0=clear, 1=cloudy, 2=rain (hourly precip > 0.5 mm)
    df["cuaca"] = np.where(
        df["precipitation"] > 0.5, 2,
        np.where(df["precipitation"] > 0.1, 1, 0)
    )

    df = generate_ph(df)

    # Rename temperature; add humidity (air, 0–1) and light_intensity (W/m²) for hydroponic lettuce
    df["suhu"] = df["temperature_2m"]
    df["humidity"] = (df["relative_humidity_2m"].fillna(70) / 100).clip(0, 1)
    df["light_intensity"] = df["shortwave_radiation"].fillna(0)

    # Derived features
    df = add_cyclical_time(df)
    df = add_wind_direction_cyclical(df)
    df = add_temp_lags(df)

    # Fill NaN (some API vars may be missing in older years)
    df["dewpoint_2m"] = df["dewpoint_2m"].fillna(df["suhu"] - 2)
    df["surface_pressure"] = df["surface_pressure"].fillna(1013)
    df["cloud_cover"] = df["cloud_cover"].fillna(50)
    df["wind_speed_10m"] = df["wind_speed_10m"].fillna(0)
    df["weather_code"] = df["weather_code"].fillna(0)
    df["wind_dir_sin"] = df["wind_dir_sin"].fillna(0)
    df["wind_dir_cos"] = df["wind_dir_cos"].fillna(0)

    # Column order for CSV (hydroponic: no soil moisture; humidity + light_intensity; precipitation for forecast)
    COLS = [
        "time", "suhu", "cuaca", "humidity", "light_intensity", "ph", "precipitation",
        "dewpoint_2m", "surface_pressure", "cloud_cover", "weather_code",
        "wind_speed_10m", "wind_dir_sin", "wind_dir_cos",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "temp_lag_24", "temp_lag_168",
    ]
    final = df[COLS].copy()

    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved dataset to {OUTPUT_PATH}")
    print(f"Total rows: {len(final)} (~{len(final)/24:.0f} days)")
    print(f"Columns: {list(final.columns)}")


if __name__ == "__main__":
    build_dataset()

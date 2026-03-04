"""
Config for Main_model pipeline: hourly data, 1-week forecast.

- INPUT_LEN = 168  (1 week in hours)
- PRED_LEN = 168   (1 week forecast)

Expanded features from fetch_openmeteo_historical: suhu, cuaca, kelembapan, ph,
dewpoint_2m, surface_pressure, cloud_cover, shortwave_radiation, wind_speed_10m,
wind_dir_sin, wind_dir_cos, hour_sin, hour_cos, doy_sin, doy_cos,
temp_lag_24, temp_lag_168
"""
# All columns in historical_environment.csv (excluding time)
ALL_FEATURES = [
    "suhu", "cuaca", "kelembapan", "ph",
    "dewpoint_2m", "surface_pressure", "cloud_cover", "shortwave_radiation",
    "wind_speed_10m", "wind_dir_sin", "wind_dir_cos",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "temp_lag_24", "temp_lag_168",
]
# Columns to save in prediction_forecast.csv (main outputs for lettuce)
FEATURES_FORECAST = ["suhu", "cuaca", "kelembapan", "ph"]

INPUT_LEN = 168   # 1 week of hourly data
PRED_LEN = 168    # 1 week forecast

"""
Config for Main_model pipeline: hourly data, 1-month forecast.
Hydroponic red leaf lettuce: Temperature, Humidity, Weather, Light intensity, pH (no soil moisture).
"""
# All columns in historical_environment.csv (excluding time)
ALL_FEATURES = [
    "suhu", "humidity", "light_intensity", "ph", "precipitation",
    "dewpoint_2m", "surface_pressure", "cloud_cover",
    "wind_speed_10m", "wind_dir_sin", "wind_dir_cos",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "temp_lag_24", "temp_lag_168",
]
# Columns to save in prediction_forecast.csv (cuaca derived from predicted precipitation)
FEATURES_FORECAST = ["suhu", "cuaca", "humidity", "light_intensity", "ph", "precipitation"]

INPUT_LEN = 168   # 1 week of hourly data
PRED_LEN = 720    # 1 month (30 days) forecast

# --- Training: extra MSE weight on suhu (channel 0) vs other variables ---
SUHU_LOSS_WEIGHT = 2.0  # 1.0 = uniform; >1 stresses temperature in the loss

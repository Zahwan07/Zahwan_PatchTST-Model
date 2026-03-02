"""
Config for Main_model pipeline: hourly data, 1-month forecast.

- INPUT_LEN = 168  (1 week in hours)
- PRED_LEN = 720   (30 days = 1 month in hours)
"""
FEATURES = ["suhu", "cuaca", "kelembapan", "ph"]
# INPUT_DIM set at runtime by preprocessor (3 continuous + n_cuaca one-hot = 6)

INPUT_LEN = 168   # 1 week of hourly data
PRED_LEN = 168    # 30 days = 1 month hourly forecast

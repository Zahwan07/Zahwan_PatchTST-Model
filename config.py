"""
Configuration for PatchTST plant growth prediction (hydroponic red leaf lettuce).
Model predicts from: temperature, weather, humidity, light intensity, pH.
"""
# Feature column names (must match CSV columns)
FEATURES = ["suhu", "cuaca", "humidity", "light_intensity", "ph"]
INPUT_DIM = len(FEATURES)  # 5

# Sequence lengths (legacy / other scripts)
INPUT_LEN = 24
PRED_LEN = 7

# Red leaf lettuce (hydroponic) — optimal ranges for recommendations
LETTUCE_TEMP_OPTIMAL = (18, 24)             # °C
LETTUCE_HUMIDITY_OPTIMAL = (0.50, 0.70)     # air humidity 50–70%
LETTUCE_LIGHT_OPTIMAL = (150, 600)          # W/m² (6–8h sun or 12–16h LED equivalent)
LETTUCE_PH_OPTIMAL = (6.0, 7.0)             # hydroponic solution pH

"""
Configuration for PatchTST plant growth prediction (final college project).
Model predicts plant growth from these 4 input parameters:
  - weather (cuaca)
  - temperature (suhu)
  - soil moisture (kelembapan)
  - plant pH (ph)
"""
# Feature column names (must match CSV columns)
FEATURES = ["suhu", "cuaca", "kelembapan", "ph"]
INPUT_DIM = len(FEATURES)  # 4

# Sequence lengths
INPUT_LEN = 24   # past timesteps per sample
PRED_LEN = 7     # forecast 7 days (1 week) ahead

# Red leaf lettuce (Lactuca sativa L.) — optimal ranges for soil moisture & pH
# See data/lettuce_reference.md for sources and details.
LETTUCE_KELEMBAPAN_OPTIMAL = (0.35, 0.65)   # fraction 0–1; ideal band 0.55–0.60
LETTUCE_PH_OPTIMAL = (6.0, 7.0)             # soil/substrate pH; ideal ~6.5

# Use official PatchTST from PatchTST-main (patching + RevIN) if folder exists; else simple Transformer
USE_OFFICIAL_PATCHTST = True
# Official PatchTST only: patch_len and stride (seq_len=INPUT_LEN, pred_len=PRED_LEN)
PATCH_LEN = 4
STRIDE = 2

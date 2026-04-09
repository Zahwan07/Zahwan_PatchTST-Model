"""
Evaluate saved Main_model checkpoint without training.

Loads: Main_model/patchtst_model_exp.pth + Main_model/preprocessor_exp.joblib
Uses the same data split as train2.py (SEED=42, 80/20 train/val).
Reports scaled MAE/MSE/RMSE/SMAPE and denormalized per-variable MAE/RMSE.

Run from project root:  python Main_model/eval_exp.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.patchtst_official import PatchTST_Official
from utils import compute_metrics
from Main_model.config_exp import INPUT_LEN, PRED_LEN
from Main_model.preprocess import create_dataset

SEED = 42
OUT_DIR = os.path.join(ROOT, "Main_model")
MODEL_PATH = os.path.join(OUT_DIR, "patchtst_model_exp.pth")
PREPROCESSOR_PATH = os.path.join(OUT_DIR, "preprocessor_exp.joblib")
DATA_PATH = os.path.join(ROOT, "data", "historical_environment.csv")

METRIC_CONT_COLS = [0, 1, 2, 3, 4]
MAPE_COLS = [0, 1, 3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

preprocessor = joblib.load(PREPROCESSOR_PATH)
INPUT_DIM = preprocessor.n_features_in_

df = pd.read_csv(DATA_PATH)
data_scaled = preprocessor.transform(df)
X, Y = create_dataset(data_scaled, input_len=INPUT_LEN, pred_len=PRED_LEN)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

torch.manual_seed(SEED)
np.random.seed(SEED)
n = len(X_tensor)
idx = torch.randperm(n)
train_end = int(0.8 * n)
X_val = X_tensor[idx[train_end:]]
Y_val = Y_tensor[idx[train_end:]]
print(f"Validation samples: {len(X_val)} (same 80/20 split as train2.py, seed={SEED})")

BATCH_SIZE = 256 if torch.cuda.is_available() else 64
val_loader = DataLoader(
    TensorDataset(X_val, Y_val),
    batch_size=BATCH_SIZE,
)

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
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

criterion_mse = nn.MSELoss()

val_loss_sum = 0.0
val_preds, val_targets = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        val_loss_sum += criterion_mse(out, yb).item()
        val_preds.append(out)
        val_targets.append(yb)

val_out = torch.cat(val_preds, dim=0)
Y_val_cat = torch.cat(val_targets, dim=0)
val_loss_avg = val_loss_sum / len(val_loader)

metrics = compute_metrics(
    val_out, Y_val_cat,
    continuous_cols=METRIC_CONT_COLS,
    mape_cols=MAPE_COLS,
)

cont_names = preprocessor.continuous_cols_
vo = val_out[:, :, :INPUT_DIM].reshape(-1, INPUT_DIM).detach().cpu().numpy()
yo = Y_val_cat[:, :, :INPUT_DIM].reshape(-1, INPUT_DIM).detach().cpu().numpy()
vo_raw = preprocessor.scaler_.inverse_transform(vo)
yo_raw = preprocessor.scaler_.inverse_transform(yo)


def _precip_to_cuaca(p):
    return np.where(p > 0.5, 2, np.where(p > 0.1, 1, 0))


pi = cont_names.index("precipitation") if "precipitation" in cont_names else None
if pi is not None:
    cuaca_match = (_precip_to_cuaca(vo_raw[:, pi]) == _precip_to_cuaca(yo_raw[:, pi])).mean() * 100.0
else:
    cuaca_match = float("nan")

print("\n--- Evaluation (saved model, validation set) ---")
print(f"Model: {MODEL_PATH}")
print(f"Loss (MSE, scaled): {val_loss_avg:.6f}")
print(f"MAE:  {metrics['MAE']:.4f}  (scaled, channels 0–4)")
print(f"MSE:  {metrics['MSE']:.6f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"MAPE: {metrics['MAPE']:.2f}%  (SMAPE on suhu, humidity, ph — scaled)")
print("\n--- Per-variable MAE / RMSE (denormalized) ---")
for j, name in enumerate(cont_names):
    err = vo_raw[:, j] - yo_raw[:, j]
    print(f"  {name}: MAE={np.abs(err).mean():.4f}, RMSE={np.sqrt((err ** 2).mean()):.4f}")
if pi is not None:
    print(f"\nDerived cuaca match (from precip): {cuaca_match:.1f}%")

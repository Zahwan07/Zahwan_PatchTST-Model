"""
Main_model training: same model and pipeline as train.py, but uses
data/historical_environment.csv (Open-Meteo) instead of fusion (internet + realtime).

Weather: model predicts precipitation (continuous) only; cuaca for display is derived
from precip thresholds (same as fetch_openmeteo_historical). No separate cuaca head.

Run from project root:  python Main_model/train2.py

Saves: Main_model/patchtst_model_exp.pth  and  Main_model/preprocessor_exp.joblib
(does not overwrite the main model or scaler.)
"""
import os
import sys

# Ensure project root is on path so we can import model, config, utils
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
import pandas as pd
import random
import numpy as np
from model.patchtst_official import PatchTST_Official
from utils import compute_metrics
from Main_model.config_exp import INPUT_LEN, PRED_LEN, SUHU_LOSS_WEIGHT
from Main_model.preprocess import prepare_data, create_dataset
# from Main_model.suhu_bias_calib import compute_suhu_hourly_bias_c, save_suhu_hourly_bias_c

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = os.path.join(ROOT, "Main_model")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Suhu channel MSE weight: {SUHU_LOSS_WEIGHT} (config_exp.py)")

print("Main_model: loading data/historical_environment.csv (all continuous; weather = precipitation) ...")
data_scaled, preprocessor = prepare_data()
INPUT_DIM = preprocessor.n_features_in_
X, Y = create_dataset(data_scaled, input_len=INPUT_LEN, pred_len=PRED_LEN)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Train / validation split (80% / 20%)
n = len(X_tensor)
idx = torch.randperm(n)
train_end = int(0.8 * n)
X_train, X_val = X_tensor[idx[:train_end]], X_tensor[idx[train_end:]]
Y_train, Y_val = Y_tensor[idx[:train_end]], Y_tensor[idx[train_end:]]
print(f"Samples: train={len(X_train)}, val={len(X_val)}")

# Mini-batch training to avoid OOM (full batch ~38 GB for attention)
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
train_ds = TensorDataset(X_train, Y_train)
val_ds = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Official PatchTST with patching (168→720, paper architecture)
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
channel_w = torch.ones(INPUT_DIM, device=device, dtype=torch.float32)
channel_w[0] = SUHU_LOSS_WEIGHT


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, ch_w: torch.Tensor) -> torch.Tensor:
    """Channel-weighted MSE; normalized by mean(ch_w) so scale matches plain MSE when weights are uniform."""
    d2 = (pred - target) ** 2
    return (d2 * ch_w.view(1, 1, -1)).mean() / ch_w.mean()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Core continuous indices for metrics (suhu, humidity, light, ph, precipitation)
METRIC_CONT_COLS = [0, 1, 2, 3, 4]
MAPE_COLS = [0, 1, 3]

EPOCHS = 300
PATIENCE = 25
best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_batches = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = weighted_mse(output, yb, channel_w)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
        train_batches += 1
    train_loss_avg = train_loss_sum / train_batches

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        model.eval()
        val_loss_sum = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_out = model(xb)
                val_loss_sum += weighted_mse(val_out, yb, channel_w).item()
                val_preds.append(val_out)
                val_targets.append(yb)
        val_out = torch.cat(val_preds, dim=0)
        Y_val_cat = torch.cat(val_targets, dim=0)
        val_loss_avg = val_loss_sum / len(val_loader)
        metrics = compute_metrics(
            val_out, Y_val_cat,
            continuous_cols=METRIC_CONT_COLS,
            mape_cols=MAPE_COLS,
        )
        print(f"Epoch {epoch:3d}, train_loss={train_loss_avg:.6f}, val_loss={val_loss_avg:.6f} | val MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.6f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 10 if epoch % 10 == 0 else 1

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    print("Restored best model (lowest validation loss).")

model.eval()
val_preds, val_targets = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        val_preds.append(model(xb))
        val_targets.append(yb)
val_out = torch.cat(val_preds, dim=0)
Y_val_cat = torch.cat(val_targets, dim=0)
final_val = weighted_mse(val_out, Y_val_cat, channel_w).item()
final_metrics = compute_metrics(
    val_out, Y_val_cat,
    continuous_cols=METRIC_CONT_COLS,
    mape_cols=MAPE_COLS,
)

# Denormalized MAE/RMSE per variable (natural units)
cont_names = preprocessor.continuous_cols_
vo = val_out[:, :, :INPUT_DIM].reshape(-1, INPUT_DIM).detach().cpu().numpy()
yo = Y_val_cat[:, :, :INPUT_DIM].reshape(-1, INPUT_DIM).detach().cpu().numpy()
vo_raw = preprocessor.scaler_.inverse_transform(vo)
yo_raw = preprocessor.scaler_.inverse_transform(yo)

# Derived cuaca from precipitation (mm/h), same thresholds as fetch_openmeteo_historical
def _precip_to_cuaca(p):
    return np.where(p > 0.5, 2, np.where(p > 0.1, 1, 0))

pi = cont_names.index("precipitation") if "precipitation" in cont_names else None
if pi is not None:
    pp_raw = vo_raw[:, pi]
    tp_raw = yo_raw[:, pi]
    cuaca_derived_acc = (_precip_to_cuaca(pp_raw) == _precip_to_cuaca(tp_raw)).mean() * 100.0
else:
    cuaca_derived_acc = float("nan")

print("\n--- Validation metrics (best model) ---")
print(f"Loss (weighted MSE, scaled space; suhu×{SUHU_LOSS_WEIGHT}): {final_val:.6f}")
print(f"MAE:  {final_metrics['MAE']:.4f}  (scaled, first 5 continuous channels)")
print(f"MSE:  {final_metrics['MSE']:.6f}")
print(f"RMSE: {final_metrics['RMSE']:.4f}")
print(f"MAPE: {final_metrics['MAPE']:.2f}%  (SMAPE on suhu, humidity, ph — scaled space)")
print("\n--- Per-variable MAE / RMSE (denormalized, validation) ---")
for j, name in enumerate(cont_names):
    err = vo_raw[:, j] - yo_raw[:, j]
    mae_v = np.abs(err).mean()
    rmse_v = np.sqrt((err ** 2).mean())
    print(f"  {name}: MAE={mae_v:.4f}, RMSE={rmse_v:.4f}")
if pi is not None:
    print(f"\nWeather (derived cuaca from precip vs target): match rate {cuaca_derived_acc:.1f}%  (thresholds: >0.5 mm rain, >0.1 mm cloudy)")

# Hourly suhu bias calibration disabled per adviser request (keep commented for easy re-enable later).
# val_indices = idx[train_end:]
# df_times = pd.read_csv(os.path.join(ROOT, "data", "historical_environment.csv"), usecols=["time"])["time"]
# suhu_bias = compute_suhu_hourly_bias_c(
#     val_out, Y_val_cat, val_indices, df_times, INPUT_LEN, PRED_LEN, preprocessor
# )
# bias_path = save_suhu_hourly_bias_c(suhu_bias, OUT_DIR)
# print(f"\nSaved hourly suhu bias (°C, add to pred): {bias_path}")
# print(f"  bias[0..23h]: {np.round(suhu_bias, 3)}")

model_path = os.path.join(OUT_DIR, "patchtst_model_exp.pth")
preprocessor_path = os.path.join(OUT_DIR, "preprocessor_exp.joblib")
torch.save(model.state_dict(), model_path)
joblib.dump(preprocessor, preprocessor_path)
print(f"\nSaved: {model_path}")
print(f"Saved: {preprocessor_path}")

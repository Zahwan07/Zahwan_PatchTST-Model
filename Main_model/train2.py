"""
Main_model training: same model and pipeline as train.py, but uses
data/historical_environment.csv (Open-Meteo) instead of fusion (internet + realtime).

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
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
import random
import numpy as np
from model.patchtst_official import PatchTST_Official
from utils import compute_metrics
from Main_model.config_exp import INPUT_LEN, PRED_LEN
from Main_model.preprocess import prepare_data, create_dataset

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = os.path.join(ROOT, "Main_model")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Main_model: loading data/historical_environment.csv (cuaca=one-hot) ...")
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
# Larger batch on GPU = fewer steps per epoch = faster training
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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

EPOCHS = 300
PATIENCE = 50
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
        loss = criterion(output, yb)
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
                val_loss_sum += criterion(val_out, yb).item()
                val_preds.append(val_out)
                val_targets.append(yb)
        val_out = torch.cat(val_preds, dim=0)
        Y_val_cat = torch.cat(val_targets, dim=0)
        val_loss_avg = val_loss_sum / len(val_loader)
        metrics = compute_metrics(val_out, Y_val_cat, continuous_cols=[0, 1, 2])
        print(f"Epoch {epoch:3d}, train_loss={train_loss_avg:.6f}, val_loss={val_loss_avg:.6f} | val MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.6f}, MAPE={metrics['MAPE']:.2f}%")

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
final_val = criterion(val_out, Y_val_cat)
final_metrics = compute_metrics(val_out, Y_val_cat, continuous_cols=[0, 1, 2])

print("\n--- Validation metrics (best model) ---")
print(f"Loss (MSE): {final_val.item():.6f}")
print(f"MAE:  {final_metrics['MAE']:.4f}")
print(f"MSE: {final_metrics['MSE']:.6f}")
print(f"MAPE: {final_metrics['MAPE']:.2f}%")

model_path = os.path.join(OUT_DIR, "patchtst_model_exp.pth")
preprocessor_path = os.path.join(OUT_DIR, "preprocessor_exp.joblib")
torch.save(model.state_dict(), model_path)
joblib.dump(preprocessor, preprocessor_path)
print(f"Saved: {model_path}")
print(f"Saved: {preprocessor_path}")

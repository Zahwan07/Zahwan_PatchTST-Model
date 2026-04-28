"""
Train 5 PatchTST regression models (Multi_Model):
  suhu, humidity, light_intensity, ph, precipitation

Cuaca classifier is trained separately (see Multi_Model/train_cuaca.py).

Artifacts are saved to: Main_model/Multi_Model/artifacts/

Run from project root:
  python Multi_Model/train_multi.py
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.patchtst_official import PatchTST_Official
from utils import compute_metrics
from Multi_Model.datasets_multi import load_historical_df, build_regression_dataset
from Multi_Model.multi_config import (
    INPUT_LEN,
    PRED_LEN,
    PATCH_LEN,
    STRIDE,
    REGRESSION_LABELS,
    artifacts_for_label,
    artifacts_dir,
)


SEED = 42
EPOCHS = 300
PATIENCE = 25


def _set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_patchtst_single_label(
    label: str,
    df,
    device: torch.device,
):
    # Build dataset for label
    X, Y, preprocessor, features = build_regression_dataset(label, df=df, input_len=INPUT_LEN, pred_len=PRED_LEN)
    input_dim = X.shape[-1]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    n = len(X_tensor)
    idx = torch.randperm(n)
    train_end = int(0.8 * n)
    X_train, X_val = X_tensor[idx[:train_end]], X_tensor[idx[train_end:]]
    Y_train, Y_val = Y_tensor[idx[:train_end]], Y_tensor[idx[train_end:]]

    batch_size = 256 if torch.cuda.is_available() else 64
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

    model = PatchTST_Official(
        input_dim=input_dim,
        input_len=INPUT_LEN,
        pred_len=PRED_LEN,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        d_model=128,
        n_heads=16,
        num_layers=3,
        dropout=0.2,
        revin=True,
    ).to(device)

    # Channel weights: focus ONLY on target (index 0). Others are zero.
    channel_w = torch.zeros(input_dim, device=device, dtype=torch.float32)
    channel_w[0] = 1.0

    def weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d2 = (pred - target) ** 2
        # mean over batch/time/features; only target channel contributes
        return (d2 * channel_w.view(1, 1, -1)).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = weighted_mse(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            preds, targets = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_loss_sum += float(weighted_mse(out, yb).item())
                    val_batches += 1
                    preds.append(out[:, :, :1])     # only target channel
                    targets.append(yb[:, :, :1])
            val_loss = val_loss_sum / max(val_batches, 1)
            pred_cat = torch.cat(preds, dim=0)
            targ_cat = torch.cat(targets, dim=0)
            metrics = compute_metrics(pred_cat, targ_cat, continuous_cols=[0], mape_cols=[0])
            print(
                f"[{label}] epoch {epoch:3d} | train={train_loss:.6f} val={val_loss:.6f} "
                f"| MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} MAPE={metrics['MAPE']:.2f}%"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 10
                if epochs_no_improve >= PATIENCE:
                    print(f"[{label}] early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, preprocessor, features


def main():
    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"INPUT_LEN={INPUT_LEN}, PRED_LEN={PRED_LEN}, patch_len={PATCH_LEN}, stride={STRIDE}")

    df = load_historical_df()

    out_dir = artifacts_dir(ROOT)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Train 5 regression models ----
    for label in REGRESSION_LABELS:
        print("\n" + "=" * 72)
        print(f"Training PatchTST regression model for label: {label}")
        print("=" * 72)
        model, preprocessor, features = _train_patchtst_single_label(label, df=df, device=device)

        art = artifacts_for_label(ROOT, label)
        torch.save(model.state_dict(), art.model_path)
        joblib.dump(preprocessor, art.preprocessor_path)
        print(f"[{label}] saved model: {art.model_path}")
        print(f"[{label}] saved preprocessor: {art.preprocessor_path}")
        print(f"[{label}] features: {features}")

    print("\nDone training 5 regression models.")
    print("Train cuaca separately: python Multi_Model/train_cuaca.py")


if __name__ == "__main__":
    main()


"""
Recompute hourly suhu bias from validation set without retraining.

Uses same split as train2 (seed=42) and saved model + preprocessor.

Run from project root:  python Main_model/calibrate_suhu_bias.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.patchtst_official import PatchTST_Official
from Main_model.config_exp import INPUT_LEN, PRED_LEN
from Main_model.preprocess import prepare_data, create_dataset
from Main_model.suhu_bias_calib import compute_suhu_hourly_bias_c, save_suhu_hourly_bias_c

SEED = 42
OUT_DIR = os.path.join(ROOT, "Main_model")
MODEL_PATH = os.path.join(OUT_DIR, "patchtst_model_exp.pth")
PREPROCESSOR_PATH = os.path.join(OUT_DIR, "preprocessor_exp.joblib")
DATA_PATH = os.path.join(ROOT, "data", "historical_environment.csv")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import joblib

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    INPUT_DIM = preprocessor.n_features_in_

    data_scaled, _ = prepare_data()
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
    val_indices = idx[train_end:]

    batch_size = 256 if torch.cuda.is_available() else 64
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

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

    val_preds, val_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_preds.append(model(xb))
            val_targets.append(yb)
    val_out = torch.cat(val_preds, dim=0)
    Y_val_cat = torch.cat(val_targets, dim=0)

    df_times = pd.read_csv(DATA_PATH, usecols=["time"])["time"]
    bias = compute_suhu_hourly_bias_c(
        val_out, Y_val_cat, val_indices, df_times, INPUT_LEN, PRED_LEN, preprocessor
    )
    path = save_suhu_hourly_bias_c(bias, OUT_DIR)
    print(f"Saved: {path}")
    print(f"bias °C by hour 0-23:\n{np.round(bias, 4)}")


if __name__ == "__main__":
    main()

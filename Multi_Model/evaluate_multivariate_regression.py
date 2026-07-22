"""
evaluate_multivariate_regression.py

Skrip evaluasi model MultivariatePatchTST (Fusion Head Regression)
yang dilatih oleh Multi_Model/train_multivariate_regression.py.

Metrik yang ditampilkan per label:
  - MAE  (Mean Absolute Error)
  - MSE  (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - SMAPE (Symmetric Mean Absolute Percentage Error)

Evaluasi dilakukan pada SET VALIDASI (80/20 split, seed identik dengan training)
sehingga hasilnya mencerminkan performa generalisasi model, bukan performa di data training.

Artifacts dibaca dari: results_new/
  - multivar_patchtst_<label>.pth
  - multivar_preprocessor_<label>.joblib

Run dari project root:
  python Multi_Model/evaluate_multivariate_regression.py
  python Multi_Model/evaluate_multivariate_regression.py --labels suhu humidity
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
    FEATURES_BY_LABEL,
)

# ── Konstanta (identik dengan train_multivariate_regression.py) ──────────────
SEED         = 42
RESULTS_DIR  = os.path.join(ROOT, "results_new")
TARGET_LABELS = ["suhu", "humidity", "light_intensity", "precipitation"]


# ── Dataset (identik dengan train_multivariate_regression.py) ────────────────
class _IndexedWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.x = x
        self.y = y
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        return (
            torch.tensor(self.x[j], dtype=torch.float32),
            torch.tensor(self.y[j], dtype=torch.float32),
        )


def _train_val_indices(n: int, train_fraction: float = 0.8, seed: int = SEED):
    """Identik dengan train_multivariate_regression.py -- HARUS sama agar val set tidak bocor."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_end = int(train_fraction * n)
    return perm[:train_end], perm[train_end:]


# ── Definisi Model (identik dengan train_multivariate_regression.py) ─────────
class MultivariatePatchTST(nn.Module):
    """
    Harus identik secara arsitektur dengan kelas di train_multivariate_regression.py
    agar load_state_dict() berhasil.
    """

    def __init__(
        self,
        input_dim: int,
        input_len: int = INPUT_LEN,
        pred_len: int = PRED_LEN,
        patch_len: int = PATCH_LEN,
        stride: int = STRIDE,
        d_model: int = 128,
        n_heads: int = 16,
        num_layers: int = 3,
        dropout: float = 0.2,
        revin: bool = True,
    ):
        super().__init__()
        self.backbone = PatchTST_Official(
            input_dim=input_dim,
            input_len=input_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            revin=revin,
        )
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)      # [B, PRED_LEN, D]
        return self.regressor(feat)  # [B, PRED_LEN, 1]


# ── Evaluasi Satu Label ───────────────────────────────────────────────────────
def _evaluate_label(label: str, df, results: list[dict]) -> None:
    model_path = os.path.join(RESULTS_DIR, f"multivar_patchtst_{label}.pth")
    pre_path   = os.path.join(RESULTS_DIR, f"multivar_preprocessor_{label}.joblib")

    if not os.path.isfile(model_path):
        print(f"  [{label}_y] SKIP -- model tidak ditemukan: {model_path}")
        print(f"             Jalankan: python Multi_Model/train_multivariate_regression.py --labels {label}")
        return
    if not os.path.isfile(pre_path):
        print(f"  [{label}_y] SKIP -- preprocessor tidak ditemukan: {pre_path}")
        return

    # Bangun dataset (sama persis dengan training)
    X, Y, preprocessor, features = build_regression_dataset(
        label, df=df, input_len=INPUT_LEN, pred_len=PRED_LEN
    )
    input_dim = X.shape[-1]
    n = len(X)

    # Slice Y ke target saja (sama dengan train_multivariate_regression.py)
    Y = Y[:, :, 0:1]  # (N, PRED_LEN, 1)

    # Ambil HANYA indeks validasi (seed harus identik dengan training)
    _, val_idx = _train_val_indices(n, train_fraction=0.8, seed=SEED)
    batch_size = 256 if torch.cuda.is_available() else 64
    val_loader = DataLoader(
        _IndexedWindowDataset(X, Y, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Muat model
    model = MultivariatePatchTST(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Jalankan inferensi pada val set
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)          # [B, PRED_LEN, 1]
            all_preds.append(out)
            all_targets.append(yb)

    pred_cat   = torch.cat(all_preds,   dim=0)   # [N_val, PRED_LEN, 1]
    target_cat = torch.cat(all_targets, dim=0)   # [N_val, PRED_LEN, 1]

    # Hitung metrik dalam scaled space
    metrics_scaled = compute_metrics(
        pred_cat, target_cat, continuous_cols=[0], mape_cols=[0]
    )

    # Inverse transform ke unit asli untuk MAE/RMSE yang interpretatif
    # Buat dummy array (N*pred_len, D) -- isi kolom 0 dari prediksi
    n_val   = pred_cat.shape[0]
    pred_np = pred_cat.cpu().numpy().reshape(-1, 1)     # (N_val*PRED_LEN, 1)
    targ_np = target_cat.cpu().numpy().reshape(-1, 1)

    dummy_pred = np.zeros((n_val * PRED_LEN, input_dim), dtype=np.float32)
    dummy_targ = np.zeros((n_val * PRED_LEN, input_dim), dtype=np.float32)
    dummy_pred[:, 0] = pred_np[:, 0]
    dummy_targ[:, 0] = targ_np[:, 0]

    pred_inv = preprocessor.inverse_transform(dummy_pred)[:, 0]
    targ_inv = preprocessor.inverse_transform(dummy_targ)[:, 0]

    pred_inv_t = torch.tensor(pred_inv, dtype=torch.float32).unsqueeze(-1)
    targ_inv_t = torch.tensor(targ_inv, dtype=torch.float32).unsqueeze(-1)
    metrics_real = compute_metrics(
        pred_inv_t, targ_inv_t, continuous_cols=[0], mape_cols=[0]
    )

    # Tampilkan
    print(f"\n  [{label}_y]  D={input_dim} fitur | val_windows={n_val}")
    print(f"  {'Metrik':<10} {'Scaled Space':>15} {'Unit Asli':>15}")
    print(f"  {'-'*42}")
    for key in ["MAE", "MSE", "RMSE", "MAPE"]:
        label_key = "SMAPE" if key == "MAPE" else key
        suffix    = "%" if key == "MAPE" else ""
        print(
            f"  {label_key:<10} "
            f"{metrics_scaled[key]:>14.6f}  "
            f"{metrics_real[key]:>13.4f}{suffix}"
        )

    results.append({
        "label":       f"{label}_y",
        "input_dim":   input_dim,
        "val_windows": n_val,
        "MAE_scaled":   round(float(metrics_scaled["MAE"]),  6),
        "MSE_scaled":   round(float(metrics_scaled["MSE"]),  6),
        "RMSE_scaled":  round(float(metrics_scaled["RMSE"]), 6),
        "SMAPE_scaled": round(float(metrics_scaled["MAPE"]), 4),
        "MAE_real":     round(float(metrics_real["MAE"]),    4),
        "MSE_real":     round(float(metrics_real["MSE"]),    4),
        "RMSE_real":    round(float(metrics_real["RMSE"]),   4),
        "SMAPE_real":   round(float(metrics_real["MAPE"]),   4),
    })


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi model MultivariatePatchTST dari results_new/."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=TARGET_LABELS,
        choices=TARGET_LABELS,
        help=f"Label yang dievaluasi. Default semua: {TARGET_LABELS}",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Evaluasi MultivariatePatchTST Regression (Fusion Head)")
    print(f"  Artifacts : {RESULTS_DIR}")
    print(f"  Split     : 80/20, seed={SEED} (identik dengan training)")
    print(f"  Metrik    : MAE, MSE, RMSE, SMAPE")
    print("=" * 60)

    df = load_historical_df()
    results: list[dict] = []

    for label in args.labels:
        _evaluate_label(label, df=df, results=results)

    if not results:
        print("\n  Tidak ada model yang berhasil dievaluasi.")
        return

    # Simpan ringkasan ke CSV
    out_csv = os.path.join(RESULTS_DIR, "evaluation_multivar.csv")
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_csv, index=False)

    print(f"\n{'='*60}")
    print("  RINGKASAN (Unit Asli)")
    print(f"{'='*60}")
    print(f"  {'Label':<22} {'MAE':>8} {'MSE':>10} {'RMSE':>8} {'SMAPE':>8}")
    print(f"  {'-'*58}")
    for row in results:
        print(
            f"  {row['label']:<22} "
            f"{row['MAE_real']:>8.4f} "
            f"{row['MSE_real']:>10.4f} "
            f"{row['RMSE_real']:>8.4f} "
            f"{row['SMAPE_real']:>7.2f}%"
        )
    print(f"\n  CSV tersimpan: {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()

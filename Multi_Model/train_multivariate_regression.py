"""
train_multivariate_regression.py

Model regresi PatchTST dengan Fusion Head untuk membuktikan integrasi multivariat
secara matematis pada model regresi (suhu_y, humidity_y, light_intensity_y, precipitation_y).

Perbedaan utama dari train_multi.py:
  - LAMA (Channel-Independent murni):
      Backbone -> output [B, pred_len, D] -> ambil indeks 0 saja (target)
      Hanya channel target yang berkontribusi ke loss.

  - BARU (Fusion Head Regression):
      Backbone -> feat [B, pred_len, D] -> Linear(D, 1) -> target_y [B, pred_len, 1]
      SEMUA channel berkontribusi ke output tunggal (target_y) melalui lapisan linier
      fusion, sehingga gradien mengalir balik ke seluruh D channel kovariat.

Konvensi Nama Variabel (interpretasi, tidak mengubah label di dataset):
  - suhu_x           = fitur suhu sebagai INPUT kovariat dari dataset
  - suhu_y           = hasil prediksi suhu sebagai OUTPUT target model
  (berlaku juga untuk humidity_x/y, light_intensity_x/y, precipitation_x/y)

Arsitektur matematika Fusion Head:
  feat_{b,t} in R^D  (output backbone per timestep, channel-independent)
  suhu_y_{b,t} = sum_{m=1}^{D} W_m * feat_{b,t,m} + b   (fusion linear)
  Loss = MSE( suhu_y_{b,t}, target_{b,t} )
  Gradient mengalir balik ke SEMUA D channel kovariat.

Config identik dengan multi_config.py:
  INPUT_LEN=168, PRED_LEN=720, PATCH_LEN=16, STRIDE=8
  d_model=128, n_heads=16, num_layers=3, dropout=0.2, revin=True
  Optimizer: Adam lr=0.001, weight_decay=1e-4
  Epochs=300, Patience=25
  Batch: 256 (GPU) / 64 (CPU)

Output artifacts disimpan ke: results_new/
  - multivar_patchtst_suhu.pth
  - multivar_preprocessor_suhu.joblib
  - (dst. untuk setiap label)

Run dari project root:
  python Multi_Model/train_multivariate_regression.py
  python Multi_Model/train_multivariate_regression.py --labels suhu humidity
"""

from __future__ import annotations

import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
)

# ── Konstanta ────────────────────────────────────────────────────────────────
SEED = 42
EPOCHS = 300
PATIENCE = 25
RESULTS_DIR = os.path.join(ROOT, "results_new")

# Label yang dilatih (ph dikecualikan karena perilakunya quasi-static)
TARGET_LABELS = ["suhu", "humidity", "light_intensity", "precipitation"]


# ── Dataset helper (identik dengan train_multi.py) ──────────────────────────
class _IndexedWindowDataset(Dataset):
    """
    Lazy-load dari numpy array; tidak menyalin seluruh array ke memori.

    Alasan perubahan dari pola train_multi.py:
    Y untuk model regresi ini hanya memerlukan 1 kolom target (indeks 0),
    sehingga Y di-slice ke (N, pred_len, 1) sebelum masuk dataset.
    Konversi ke float32 dilakukan per-item di __getitem__ (bukan sekaligus).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray):
        # Simpan referensi saja -- TIDAK menyalin/mengkonversi seluruh array
        # Konversi float32 dilakukan per-item saat __getitem__ dipanggil
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


def _set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_val_indices(n: int, train_fraction: float = 0.8, seed: int = SEED):
    """Identik dengan train_multi.py: permutasi acak sekali, split 80/20."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_end = int(train_fraction * n)
    return perm[:train_end], perm[train_end:]


# ── Model Baru: MultivariatePatchTST ────────────────────────────────────────
class MultivariatePatchTST(nn.Module):
    """
    PatchTST backbone + Cross-Channel Fusion Regression Head.

    Prinsip identik dengan CuacaPatchTST (klasifikasi) namun untuk regresi:
      - CuacaPatchTST : Linear(D, 3)  -> logits kelas cuaca (fusi ke 3 keluaran)
      - Model ini     : Linear(D, 1)  -> nilai kontinu target (fusi ke 1 keluaran)

    Aliran data (Forward Pass):
      x      : [B, INPUT_LEN, D]     <- masukan multivariat (D channel)
      backbone: channel-independent encoder (sama persis dengan train_multi.py)
      feat   : [B, PRED_LEN, D]      <- representasi laten per-channel
      regressor(feat): Linear(D, 1)  <- FUSION: semua D channel digabungkan
      output : [B, PRED_LEN, 1]      <- prediksi tunggal target_y

    Persamaan Fusion Head (per timestep t):
      target_y_{b,t} = sum_{m=1}^{D} W_m * feat_{b,t,m} + b

    Implikasi Backpropagation:
      MSE loss dihitung pada target_y (skalar per timestep).
      Gradien mengalir balik melalui W_m ke SEMUA D channel kovariat,
      membuktikan kontribusi multivariat secara matematis.

    Konvensi nama (interpretasi):
      suhu_x = suhu sebagai fitur input kovariat
      suhu_y = suhu sebagai output target yang diprediksi
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
        # Backbone identik dengan train_multi.py
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
        # Fusion/regression head: gabungkan semua D channel -> 1 target
        # Analog dengan CuacaPatchTST: nn.Linear(input_dim, 3)
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feat: [B, pred_len, input_dim]  (channel-independent dalam backbone)
        feat = self.backbone(x)
        # target_y: [B, pred_len, 1]  (cross-channel fusion, analog cuaca classifier)
        return self.regressor(feat)


# ── Fungsi Training Satu Label ───────────────────────────────────────────────
def _train_single_label(
    label: str,
    df,
    device: torch.device,
    out_dir: str,
) -> None:
    """
    Latih MultivariatePatchTST untuk satu label regresi dan simpan artifacts.

    Berbeda dengan train_multi.py:
      - Model: MultivariatePatchTST (backbone + Linear(D, 1) fusion head)
      - Loss : MSE antara target_y_{pred} dan target_{ground_truth, index=0}
      - Seluruh D channel memiliki gradien karena melewati fusion head
    """
    print(f"\n{'='*72}")
    print(f"  Training MultivariatePatchTST Regression | label: {label}_y")
    print(f"  ('{label}_x' adalah fitur input, '{label}_y' adalah output target)")
    print(f"{'='*72}")

    X, Y, preprocessor, features = build_regression_dataset(
        label, df=df, input_len=INPUT_LEN, pred_len=PRED_LEN
    )
    input_dim = X.shape[-1]
    n = len(X)

    print(f"  Fitur input ({label}_x dan kovariat): {features}")
    print(f"  input_dim (D) = {input_dim}")
    print(f"  Total windows  = {n}")

    # Pre-slice Y ke kolom target saja (indeks 0) untuk menghemat memori.
    # Y awalnya (N, pred_len, D); setelah slice menjadi (N, pred_len, 1).
    # Hemat memori: dari ~1.78 GiB (D=12) menjadi ~150 MB (D=1).
    Y = Y[:, :, 0:1]  # shape: (N, PRED_LEN, 1)

    train_idx, val_idx = _train_val_indices(n, train_fraction=0.8, seed=SEED)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    batch_size = 256 if torch.cuda.is_available() else 64
    train_loader = DataLoader(
        _IndexedWindowDataset(X, Y, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        _IndexedWindowDataset(X, Y, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = MultivariatePatchTST(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"\n  [Arsitektur] Fusion Head: Linear({input_dim}, 1)")
    print(f"  Persamaan: {label}_y_t = sum_m(W_m * feat_t_m) + b")
    print(f"  Batch={batch_size}, Epochs={EPOCHS}, Patience={PATIENCE}\n")

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # out: [B, pred_len, 1]  (target_y setelah fusion head)
            out = model(xb)
            # yb sudah di-slice ke (B, pred_len, 1) sejak pembangunan dataset
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)

        # ── Validasi (setiap 10 epoch) ────────────────────────────────────
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            preds, targets = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)   # [B, pred_len, 1]
                    # yb sudah (B, pred_len, 1) dari dataset
                    val_loss_sum += float(criterion(out, yb).item())
                    val_batches += 1
                    preds.append(out)
                    targets.append(yb)

            val_loss = val_loss_sum / max(val_batches, 1)
            pred_cat = torch.cat(preds, dim=0)    # [N, pred_len, 1]
            targ_cat = torch.cat(targets, dim=0)  # [N, pred_len, 1]
            metrics = compute_metrics(pred_cat, targ_cat, continuous_cols=[0], mape_cols=[0])

            print(
                f"  [{label}_y] epoch {epoch:3d} | "
                f"train={train_loss:.6f} val={val_loss:.6f} | "
                f"MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
                f"SMAPE={metrics['MAPE']:.2f}%"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 10
                if epochs_no_improve >= PATIENCE:
                    print(f"  [{label}_y] early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [{label}_y] restored best model (val_loss={best_val:.6f})")

    # ── Simpan artifacts ──────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"multivar_patchtst_{label}.pth")
    pre_path   = os.path.join(out_dir, f"multivar_preprocessor_{label}.joblib")

    torch.save(model.state_dict(), model_path)
    joblib.dump(preprocessor, pre_path)

    print(f"  [{label}_y] Saved model       : {model_path}")
    print(f"  [{label}_y] Saved preprocessor: {pre_path}")
    print(f"  [{label}_y] Fitur input (D={input_dim}): {features}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Latih MultivariatePatchTST (Fusion Head Regression) "
                    "untuk pembuktian integrasi multivariat pada model regresi."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=TARGET_LABELS,
        choices=REGRESSION_LABELS,
        help=f"Label yang dilatih. Default: {TARGET_LABELS}",
    )
    args = parser.parse_args()

    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("  MultivariatePatchTST Regression Training")
    print("  (Fusion Head: Linear(D,1) -- analog dengan CuacaPatchTST)")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  INPUT_LEN    : {INPUT_LEN} jam  | PRED_LEN: {PRED_LEN} jam")
    print(f"  PATCH_LEN    : {PATCH_LEN}      | STRIDE  : {STRIDE}")
    print(f"  Labels       : {args.labels}")
    print(f"  Output dir   : {RESULTS_DIR}")
    print(f"\n  Konvensi nama:")
    print(f"    <label>_x = fitur input dari dataset (kovariat)")
    print(f"    <label>_y = output prediksi target model")
    print("=" * 72)

    df = load_historical_df()

    for label in args.labels:
        _train_single_label(label, df=df, device=device, out_dir=RESULTS_DIR)

    print(f"\n{'='*72}")
    print(f"  Selesai. Semua artifacts tersimpan di: {RESULTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()

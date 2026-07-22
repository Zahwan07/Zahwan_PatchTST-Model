"""
ablation_multivar.py

Ablation Study: Kontribusi Multivariabel pada Model MultivariatePatchTST
(Fusion Head Regression) yang tersimpan di results_new/

Tujuan:
  Membuktikan secara empiris bahwa variabel-variabel kovariat yang menjadi
  fitur masukan model benar-benar berkontribusi terhadap prediksi target
  (suhu_y, humidity_y, light_intensity_y, precipitation_y).

Perbedaan dari ablation_suhu.py (model lama):
  - Model lama (train_multi.py)    : PatchTST_Official -> output [B, 720, D] -> ambil indeks 0
    - Ablation selalu 0 karena channel-independent murni, gradien kovariat = 0
  - Model baru (train_multivariate): MultivariatePatchTST -> Linear(D,1) -> [B, 720, 1]
    - Ablation bermakna karena Fusion Head menggabungkan semua D channel

Metode (Channel Ablation):
  1. Inferensi baseline: semua D fitur aktif -> out_baseline [B, 720, 1]
  2. Untuk setiap kovariat m (indeks 1 sampai D-1):
     - Set kolom m = 0 sepanjang 168 jam window input
     - Inferensi ulang -> out_ablated [B, 720, 1]
     - MAE perubahan = |out_ablated - out_baseline|.mean()
  3. Semakin tinggi MAE perubahan, semakin besar kontribusi kovariat tersebut.

Interpretasi:
  Karena Fusion Head Linear(D,1) menggabungkan semua D channel, menolkan
  satu kovariat akan mengubah output -- membuktikan bahwa kovariat tersebut
  benar-benar dipakai model.

Run dari project root:
  python ablation_multivar.py
  python ablation_multivar.py --labels suhu humidity
  python ablation_multivar.py --labels suhu --top 5

Requirements:
  - results_new/multivar_patchtst_<label>.pth
  - results_new/multivar_preprocessor_<label>.joblib
  (Jalankan: python Multi_Model/train_multivariate_regression.py)
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

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.patchtst_official import PatchTST_Official
from Multi_Model.multi_config import (
    INPUT_LEN,
    PRED_LEN,
    PATCH_LEN,
    STRIDE,
    FEATURES_BY_LABEL,
)

# ── Konstanta ────────────────────────────────────────────────────────────────
RESULTS_DIR   = os.path.join(ROOT, "results_new")
DATA_PATH     = os.path.join(ROOT, "data", "historical_environment.csv")
TARGET_LABELS = ["suhu", "humidity", "light_intensity", "precipitation"]


# ── Definisi Model (identik dengan train_multivariate_regression.py) ─────────
class MultivariatePatchTST(nn.Module):
    """
    Harus identik dengan kelas di train_multivariate_regression.py
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


# ── Helper ───────────────────────────────────────────────────────────────────
def _load_model(label: str, input_dim: int) -> MultivariatePatchTST:
    model_path = os.path.join(RESULTS_DIR, f"multivar_patchtst_{label}.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}\n"
            f"Jalankan: python Multi_Model/train_multivariate_regression.py --labels {label}"
        )
    model = MultivariatePatchTST(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def _run_inference(model: MultivariatePatchTST, X_tensor: torch.Tensor) -> np.ndarray:
    """Jalankan inferensi, return prediksi shape (PRED_LEN,) dalam scaled space."""
    with torch.no_grad():
        # out: [1, PRED_LEN, 1]
        out = model(X_tensor).squeeze().numpy()  # shape: (PRED_LEN,)
    return out


# ── Ablation Satu Label ───────────────────────────────────────────────────────
def _ablate_label(label: str, df: pd.DataFrame, top_n: int | None) -> list[dict]:
    """
    Jalankan channel ablation untuk satu label.
    Return list of dicts: {feat_name, mae_change, max_change}.
    """
    pre_path = os.path.join(RESULTS_DIR, f"multivar_preprocessor_{label}.joblib")
    if not os.path.isfile(pre_path):
        print(f"  [{label}_y] SKIP -- preprocessor tidak ditemukan: {pre_path}")
        return []

    pre      = joblib.load(pre_path)
    features = FEATURES_BY_LABEL[label]
    input_dim = len(features)
    target_col = features[0]   # indeks 0 = target (_x)
    covariates = features[1:]  # indeks 1..D-1 = kovariat

    print(f"\n{'='*68}")
    print(f"  Ablation Study: model {label}_y")
    print(f"{'='*68}")
    print(f"  Target input  (indeks 0): '{target_col}_x'")
    print(f"  Jumlah fitur input (D)  : {input_dim}")
    print(f"  Kovariat yang diuji     : {covariates}")
    print(f"  Jendela input           : {INPUT_LEN} jam | Prediksi: {PRED_LEN} jam")

    # Siapkan window input (168 jam terakhir)
    window_df = df.tail(INPUT_LEN)
    X_scaled  = pre.transform(window_df)                           # (168, D)
    X_tensor_baseline = torch.tensor(
        X_scaled, dtype=torch.float32
    ).unsqueeze(0)                                                 # (1, 168, D)

    # Muat model
    model = _load_model(label, input_dim=input_dim)

    # Baseline: semua D fitur aktif
    pred_baseline = _run_inference(model, X_tensor_baseline)      # (720,)

    print(f"\n  [BASELINE] Prediksi {label}_y (semua {input_dim} fitur aktif):")
    print(f"    mean={pred_baseline.mean():.4f}  std={pred_baseline.std():.4f}  "
          f"min={pred_baseline.min():.4f}  max={pred_baseline.max():.4f}  (scaled)")

    # Ablation: nolkan setiap kovariat satu per satu
    print(f"\n  {'-'*66}")
    print(f"  Pengaruh Setiap Kovariat terhadap Prediksi {label}_y")
    print(f"  (MAE = rata-rata |pred_ablated - pred_baseline| sepanjang 720 jam)")
    print(f"  {'-'*66}")
    print(f"  {'Idx':<5} {'Nama Kovariat':<28} {'MAE Perubahan':>15} {'Max |Delta|':>11}")
    print(f"  {'-'*66}")

    ablation_results = []
    for feat_idx in range(1, input_dim):
        feat_name = features[feat_idx]

        X_ablated = X_scaled.copy()
        X_ablated[:, feat_idx] = 0.0   # nolkan kovariat ini sepanjang 168 jam

        X_tensor_ablated = torch.tensor(
            X_ablated, dtype=torch.float32
        ).unsqueeze(0)

        pred_ablated = _run_inference(model, X_tensor_ablated)

        diff       = np.abs(pred_ablated - pred_baseline)
        mae_change = float(diff.mean())
        max_change = float(diff.max())

        ablation_results.append({
            "label":      f"{label}_y",
            "feat_idx":   feat_idx,
            "feat_name":  feat_name,
            "mae_change": mae_change,
            "max_change": max_change,
        })
        print(f"  [{feat_idx:<3}] {feat_name:<28} {mae_change:>15.6f} {max_change:>11.6f}")

    # Urutkan dari kontribusi terbesar
    ablation_results.sort(key=lambda r: r["mae_change"], reverse=True)
    max_mae = ablation_results[0]["mae_change"] if ablation_results else 1.0

    print(f"\n  Peringkat Kontribusi Kovariat (untuk {label}_y):")
    display = ablation_results[:top_n] if top_n else ablation_results
    for rank, row in enumerate(display, 1):
        bar_len = int(row["mae_change"] / max_mae * 30) if max_mae > 0 else 0
        bar     = "|" * bar_len
        print(f"  #{rank:<3} {row['feat_name']:<28} {bar:<30} {row['mae_change']:.6f}")

    # Kesimpulan
    contributing = [r for r in ablation_results if r["mae_change"] > 1e-6]
    print(f"\n  Kesimpulan [{label}_y]:")
    print(f"    {len(contributing)} dari {len(ablation_results)} kovariat")
    if len(contributing) == len(ablation_results):
        print(f"    terbukti memengaruhi prediksi {label}_y (MAE > 0).")
        print(f"    Seluruh {input_dim} fitur input benar-benar dipakai oleh model.")
    elif len(contributing) > 0:
        contrib_names = [r["feat_name"] for r in contributing]
        print(f"    memiliki pengaruh terukur: {contrib_names}")
    else:
        print(f"    -- tidak ada perubahan terdeteksi.")

    return ablation_results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ablation study untuk MultivariatePatchTST dari results_new/."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=TARGET_LABELS,
        choices=TARGET_LABELS,
        help=f"Label yang diuji. Default semua: {TARGET_LABELS}",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Tampilkan hanya N kovariat dengan kontribusi terbesar (default: semua).",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("  Ablation Study: MultivariatePatchTST Regression (results_new/)")
    print("=" * 68)
    print(f"  Artifacts: {RESULTS_DIR}")
    print(f"  Labels   : {args.labels}")
    if args.top:
        print(f"  Top-N    : {args.top} kovariat terbesar per label")

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("time").reset_index(drop=True)

    all_results: list[dict] = []
    for label in args.labels:
        rows = _ablate_label(label, df=df, top_n=args.top)
        all_results.extend(rows)

    if not all_results:
        print("\nTidak ada hasil ablation yang berhasil.")
        return

    # Simpan hasil ke CSV
    out_csv = os.path.join(RESULTS_DIR, "ablation_results_multivar.csv")
    pd.DataFrame(all_results).to_csv(out_csv, index=False)

    # Ringkasan antar label
    print(f"\n\n{'='*68}")
    print("  RINGKASAN ABLATION SEMUA LABEL")
    print(f"  {'Label':<20} {'Kovariat Paling Berpengaruh':<28} {'MAE':>10}")
    print(f"  {'-'*65}")
    for label in args.labels:
        rows = [r for r in all_results if r["label"] == f"{label}_y"]
        if not rows:
            continue
        top_row = max(rows, key=lambda r: r["mae_change"])
        print(
            f"  {label+'_y':<20} {top_row['feat_name']:<28} "
            f"{top_row['mae_change']:>10.6f}"
        )
    print(f"\n  CSV tersimpan: {out_csv}")
    print("=" * 68)


if __name__ == "__main__":
    main()

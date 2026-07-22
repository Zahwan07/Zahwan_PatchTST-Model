"""
predict_multivariate_regression.py

Inference script untuk model MultivariatePatchTST (Fusion Head Regression)
yang dihasilkan oleh Multi_Model/train_multivariate_regression.py.

Perbedaan dari predict_multi.py:
  - Membaca artifacts dari: results_new/
  - Nama file model       : multivar_patchtst_<label>.pth
  - Nama file preprocessor: multivar_preprocessor_<label>.joblib
  - Arsitektur model      : MultivariatePatchTST (backbone + Linear(D,1) head)
      - Backbone output   : feat [B, pred_len, D]   (channel-independent)
      - Fusion head output: [B, pred_len, 1]         (cross-channel)
  - Model cuaca           : diambil dari Results/720_80-20/ (tidak berubah)
  - Output CSV            : results_new/prediction_forecast_multivar.csv

Konvensi nama variabel (interpretasi, tidak mengubah label di dataset):
  - suhu_x      = suhu sebagai fitur input kovariat
  - suhu_y      = suhu sebagai output target yang diprediksi (fusion head)
  (berlaku sama untuk humidity, light_intensity, precipitation)

Run dari project root:
  python Multi_Model/predict_multivariate_regression.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# ── Path artifacts ────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(ROOT, "data", "historical_environment.csv")
RESULTS_DIR = os.path.join(ROOT, "results_new")
OUT_CSV     = os.path.join(RESULTS_DIR, "prediction_forecast_multivar.csv")

# Artifacts cuaca tetap dari Results/720_80-20 (tidak berubah)
CUACA_ARTIFACTS_DIR = os.path.join(ROOT, "results_new")

# Label yang menggunakan MultivariatePatchTST (Fusion Head Regression)
MULTIVAR_LABELS = ["suhu", "humidity", "light_intensity", "precipitation"]


# ── Definisi Model (identik dengan train_multivariate_regression.py) ─────────
class MultivariatePatchTST(nn.Module):
    """
    PatchTST backbone + Cross-Channel Fusion Regression Head.

    Harus identik dengan kelas di train_multivariate_regression.py
    agar load_state_dict() berhasil (nama dan bentuk parameter harus sama).

    Forward:
      x    : [B, INPUT_LEN, D]   <- input multivariat
      feat : [B, PRED_LEN, D]    <- output backbone (channel-independent)
      out  : [B, PRED_LEN, 1]    <- output fusion head (cross-channel)

    Persamaan per timestep t:
      <label>_y_t = sum_{m=1}^{D} W_m * feat_{t,m} + b
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
        # Fusion head: Linear(D, 1) -- cross-channel regression
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)      # [B, PRED_LEN, D]
        return self.regressor(feat)  # [B, PRED_LEN, 1]


# ── Model Cuaca (identik dengan predict_multi.py) ─────────────────────────────
class CuacaPatchTST(nn.Module):
    """Wrapper cuaca classifier -- tidak berubah dari predict_multi.py."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = PatchTST_Official(
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
        )
        self.classifier = nn.Linear(input_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)


# ── Helper: load MultivariatePatchTST ─────────────────────────────────────────
def _load_multivar_patchtst(label: str, input_dim: int) -> MultivariatePatchTST:
    model_path = os.path.join(RESULTS_DIR, f"multivar_patchtst_{label}.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model artifact tidak ditemukan: {model_path}\n"
            f"Jalankan terlebih dahulu:\n"
            f"  python Multi_Model/train_multivariate_regression.py --labels {label}"
        )
    model = MultivariatePatchTST(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# ── Helper: load CuacaPatchTST ────────────────────────────────────────────────
def _load_cuaca_patchtst(input_dim: int) -> CuacaPatchTST:
    model_path = os.path.join(CUACA_ARTIFACTS_DIR, "patchtst_cuaca.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Cuaca model tidak ditemukan: {model_path}\n"
            f"Pastikan artifacts cuaca ada di: {CUACA_ARTIFACTS_DIR}"
        )
    model = CuacaPatchTST(input_dim=input_dim)
    payload = torch.load(model_path, map_location="cpu")
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("time").reset_index(drop=True)

    last_time = pd.to_datetime(df["time"].iloc[-1])
    pred_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1), periods=PRED_LEN, freq="h"
    )

    print("=" * 72)
    print("  Inference: MultivariatePatchTST Regression (Fusion Head)")
    print(f"  Artifacts : {RESULTS_DIR}")
    print(f"  Output CSV: {OUT_CSV}")
    print("=" * 72)

    # ── Regresi: MultivariatePatchTST (suhu_y, humidity_y, dst.) ─────────────
    preds = {}
    for label in MULTIVAR_LABELS:
        pre_path = os.path.join(RESULTS_DIR, f"multivar_preprocessor_{label}.joblib")
        if not os.path.isfile(pre_path):
            raise FileNotFoundError(
                f"Preprocessor tidak ditemukan: {pre_path}\n"
                f"Jalankan terlebih dahulu:\n"
                f"  python Multi_Model/train_multivariate_regression.py --labels {label}"
            )
        pre = joblib.load(pre_path)
        feats = FEATURES_BY_LABEL[label]

        window_df = df.tail(INPUT_LEN)
        X_scaled = pre.transform(window_df)             # (168, D)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        input_dim = X_scaled.shape[-1]

        model = _load_multivar_patchtst(label, input_dim=input_dim)

        with torch.no_grad():
            # out: [1, PRED_LEN, 1]  (hasil fusion head)
            out_scaled = model(X_tensor).squeeze(-1).numpy()  # (1, PRED_LEN)

        # Inverse transform: buat array (PRED_LEN, D) dengan hanya kolom 0 terisi
        # (kolom lain 0 -- hanya untuk keperluan inverse scaler; target di index 0)
        dummy = np.zeros((PRED_LEN, input_dim), dtype=np.float32)
        dummy[:, 0] = out_scaled[0]
        y_subset = pre.inverse_transform(dummy)          # (PRED_LEN, D)
        preds[label] = y_subset[:, 0].astype(float)

        print(
            f"  [{label}_y] D={input_dim} | mean={preds[label].mean():.4f} "
            f"min={preds[label].min():.4f} max={preds[label].max():.4f}"
        )

    # ── Cuaca Classifier (tidak berubah, dari Results/720_80-20) ─────────────
    pre_c_path = os.path.join(CUACA_ARTIFACTS_DIR, "preprocessor_cuaca.joblib")
    pre_c = joblib.load(pre_c_path)
    window_df = df.tail(INPUT_LEN)
    Xc_scaled = pre_c.transform(window_df)              # (168, D_cuaca)
    Xc_tensor = torch.tensor(Xc_scaled, dtype=torch.float32).unsqueeze(0)
    input_dim_cuaca = Xc_scaled.shape[-1]

    cuaca_model = _load_cuaca_patchtst(input_dim=input_dim_cuaca)
    with torch.no_grad():
        logits = cuaca_model(Xc_tensor)                  # (1, 720, 3)
        cuaca_pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # Align cuaca dengan hasil prediksi precipitation (sama seperti predict_multi.py)
    precip_series = np.asarray(preds["precipitation"], dtype=float)
    precip_as_cuaca = np.where(
        precip_series > 0.5, 2, np.where(precip_series > 0.1, 1, 0)
    ).astype(np.int64)
    cuaca_out = np.maximum(cuaca_pred.astype(np.int64), precip_as_cuaca)

    print(f"  [cuaca]    D={input_dim_cuaca} | distribusi kelas: "
          f"0={int((cuaca_out==0).sum())} 1={int((cuaca_out==1).sum())} 2={int((cuaca_out==2).sum())}")

    # ── Susun Output CSV ──────────────────────────────────────────────────────
    out = pd.DataFrame(
        {
            "time":            pred_times,
            "suhu_y":          preds["suhu"],
            "humidity_y":      preds["humidity"],
            "light_intensity_y": preds["light_intensity"],
            "precipitation_y": preds["precipitation"],
            "cuaca":           cuaca_out.astype(float),
        }
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"\n  Saved forecast CSV: {OUT_CSV}")
    print("\n  Preview (3 baris pertama):")
    print(out.head(3).to_string(index=False))
    print("  ...")
    print("\n  Preview (3 baris terakhir):")
    print(out.tail(3).to_string(index=False))
    print("\n" + "=" * 72)
    print("  Keterangan kolom:")
    print("    suhu_y, humidity_y, dst. = output target dari Fusion Head")
    print("    (suffix _y = prediksi target; _x = fitur input di dataset)")
    print("=" * 72)


if __name__ == "__main__":
    main()

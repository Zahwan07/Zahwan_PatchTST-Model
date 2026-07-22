"""
Ablation Study: Kontribusi Multivariabel terhadap Model Lingkungan (Suhu/Cuaca/dll)

Tujuan:
  Membuktikan secara empiris bahwa variabel-variabel kovariat yang menjadi
  fitur masukan model benar-benar berkontribusi dan saling melengkapi
  untuk menghasilkan prediksi.

Metode:
  1. Deteksi arsitektur model secara otomatis (Klasifikasi Cuaca, Regresi Baru, atau Regresi Lama).
  2. Jalankan inferensi dengan semua fitur aktif (Baseline).
  3. Jalankan inferensi dengan meniadakan (menyetel ke 0) masing-masing variabel kovariat secara bergantian.
  4. Hitung selisih prediksi:
     - Untuk Regresi: Selisih absolut nilai prediksi (MAE).
     - Untuk Klasifikasi (Cuaca): Selisih rata-rata probabilitas kelas (setelah Softmax)
       dan jumlah jam kelas prediksi bergeser.

Run dari project root:
  python ablation_suhu.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── path setup ───────────────────────────────────────────────────────────────
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

# Direktori artifacts
ARTIFACTS_DIR = os.path.join(ROOT, "results_new")

# Set LABEL ke "cuaca" atau "suhu"
LABEL = "cuaca"
DATA_PATH = os.path.join(ROOT, "data", "historical_environment.csv")


# ── Definisi Model Arsitektur ────────────────────────────────────────────────

class CuacaPatchTST(nn.Module):
    """Classifier Cuaca berbasis PatchTST."""
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


class MultivariatePatchTST(nn.Module):
    """Model Regresi Baru dengan Fusion Head Linear(D, 1)."""
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
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)      # [B, PRED_LEN, D]
        return self.regressor(feat)  # [B, PRED_LEN, 1]


def detect_and_load_model(model_path: str, input_dim: int) -> tuple[nn.Module, bool, bool]:
    """Mendeteksi arsitektur model dari state_dict dan memuat bobotnya."""
    payload = torch.load(model_path, map_location="cpu")
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    # Deteksi berdasarkan key di state_dict
    has_classifier = any(k.startswith("classifier.") for k in state_dict.keys())
    has_regressor = any(k.startswith("regressor.") for k in state_dict.keys())

    if has_classifier:
        print("  [INFO] Terdeteksi model: CuacaPatchTST (Klasifikasi Cuaca)")
        model = CuacaPatchTST(input_dim=input_dim)
    elif has_regressor:
        print("  [INFO] Terdeteksi model: MultivariatePatchTST (Regresi Baru)")
        model = MultivariatePatchTST(input_dim=input_dim)
    else:
        print("  [INFO] Terdeteksi model: PatchTST_Official (Regresi Lama)")
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
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model, has_classifier, has_regressor


def run_inference(model: nn.Module, X_tensor: torch.Tensor, has_classifier: bool, has_regressor: bool) -> np.ndarray:
    """Jalankan inferensi dan ambil outputs."""
    with torch.no_grad():
        if has_classifier:
            # Output: [1, PRED_LEN, 3] (logits)
            logits = model(X_tensor)
            return logits.squeeze(0).numpy()
        elif has_regressor:
            # Output: [1, PRED_LEN, 1]
            out = model(X_tensor)
            return out.squeeze().numpy()
        else:
            # Output: [1, PRED_LEN, input_dim] -> ambil kolom 0 (target)
            y_scaled = model(X_tensor).numpy()
            return y_scaled[0, :, 0]


def main():
    # ── 1. Cari file model & preprocessor secara fleksibel ───────────────────
    is_cuaca = (LABEL == "cuaca")
    if is_cuaca:
        model_path = os.path.join(ARTIFACTS_DIR, "patchtst_cuaca.pth")
        preprocessor_path = os.path.join(ARTIFACTS_DIR, "preprocessor_cuaca.joblib")
    else:
        model_path = os.path.join(ARTIFACTS_DIR, f"multivar_patchtst_{LABEL}.pth")
        if not os.path.isfile(model_path):
            model_path = os.path.join(ARTIFACTS_DIR, f"patchtst_{LABEL}.pth")

        preprocessor_path = os.path.join(ARTIFACTS_DIR, f"multivar_preprocessor_{LABEL}.joblib")
        if not os.path.isfile(preprocessor_path):
            preprocessor_path = os.path.join(ARTIFACTS_DIR, f"preprocessor_{LABEL}.joblib")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model artifact tidak ditemukan di: {model_path}\n"
            f"Pastikan file ada di folder: {ARTIFACTS_DIR}"
        )

    pre = joblib.load(preprocessor_path)
    features: list[str] = FEATURES_BY_LABEL[LABEL]
    input_dim = len(features)

    print("=" * 75)
    print(f"  ABLATION STUDY: Model {LABEL.upper()} ({ARTIFACTS_DIR})")
    print("=" * 75)
    print(f"  Total fitur masukan (D)       : {input_dim}")
    print(f"  Fitur masukan                 : {features}")
    print(f"  Target utama                  : '{features[0]}'")
    print(f"  Variabel kovariat             : {features[1:]}")
    print(f"  Jendela input (INPUT_LEN)     : {INPUT_LEN} jam")
    print(f"  Horison prediksi (PRED_LEN)   : {PRED_LEN} jam")
    print("=" * 75)

    # ── 2. Siapkan data window 168 jam terakhir ──────────────────────────────
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("time").reset_index(drop=True)
    window_df = df.tail(INPUT_LEN)
    X_scaled = pre.transform(window_df)          # shape: (168, D)
    X_tensor_baseline = torch.tensor(
        X_scaled, dtype=torch.float32
    ).unsqueeze(0)                               # shape: (1, 168, D)

    # ── 3. Muat model & Deteksi arsitektur ───────────────────────────────────
    model, has_classifier, has_regressor = detect_and_load_model(model_path, input_dim)

    # ── 4. Jalankan Baseline (semua fitur aktif) ─────────────────────────────
    pred_baseline = run_inference(model, X_tensor_baseline, has_classifier, has_regressor)

    if has_classifier:
        # Hitung probabilitas baseline menggunakan softmax
        exp_base = np.exp(pred_baseline - np.max(pred_baseline, axis=-1, keepdims=True))
        probs_baseline = exp_base / np.sum(exp_base, axis=-1, keepdims=True)
        classes_baseline = np.argmax(pred_baseline, axis=-1)

        print(f"\n[BASELINE] Prediksi kelas {LABEL} (total {PRED_LEN} jam):")
        classes_named = ["Cerah (0)", "Berawan (1)", "Hujan (2)"]
        for c_idx, c_name in enumerate(classes_named):
            count = np.sum(classes_baseline == c_idx)
            pct = (count / PRED_LEN) * 100
            print(f"  Kelas {c_name:<15}: {count:>4} jam ({pct:.1f}%)")
    else:
        print(f"\n[BASELINE] Prediksi {LABEL}:")
        print(f"  Mean prediksi = {pred_baseline.mean():.4f} (scaled space)")
        print(f"  Std prediksi  = {pred_baseline.std():.4f}")

    # ── 5. Ablasi Fitur Satu per Satu ────────────────────────────────────────
    print(f"\n{'-'*75}")
    if has_classifier:
        print(f"  Pengaruh Setiap Variabel Kovariat terhadap Prediksi {LABEL.upper()}")
        print("  - MAE Probabilitas = rata-rata perubahan nilai probabilitas (setelah Softmax)")
        print(f"  - Jam Berubah      = jumlah jam (dari {PRED_LEN}) yang kelas prediksinya bergeser")
        print(f"{'-'*75}")
        print(f"  {'Idx':<5} {'Nama Kovariat':<25} {'MAE Probabilitas':>18} {'Jam Berubah':>15}")
    else:
        print(f"  Pengaruh Setiap Variabel Kovariat terhadap Prediksi {LABEL.upper()}")
        print("  (MAE Perubahan = |Pred_Ablasi - Pred_Baseline|.mean() sepanjang 720 jam)")
        print(f"{'-'*75}")
        print(f"  {'Idx':<5} {'Nama Kovariat':<25} {'MAE Perubahan':>18} {'Max |Delta|':>15}")
    print(f"{'-'*75}")

    ablation_results = []

    for feat_idx in range(1, input_dim):
        feat_name = features[feat_idx]

        # Nolkan saluran fitur ke-feat_idx
        X_ablated = X_scaled.copy()
        X_ablated[:, feat_idx] = 0.0
        X_tensor_ablated = torch.tensor(
            X_ablated, dtype=torch.float32
        ).unsqueeze(0)

        pred_ablated = run_inference(model, X_tensor_ablated, has_classifier, has_regressor)

        if has_classifier:
            # Hitung probabilitas ablasi
            exp_abl = np.exp(pred_ablated - np.max(pred_ablated, axis=-1, keepdims=True))
            probs_ablated = exp_abl / np.sum(exp_abl, axis=-1, keepdims=True)
            classes_ablated = np.argmax(pred_ablated, axis=-1)

            diff = np.abs(probs_ablated - probs_baseline)
            mae_change = float(diff.mean())
            class_changes = int(np.sum(classes_baseline != classes_ablated))

            ablation_results.append((feat_name, mae_change, class_changes))
            print(f"  [{feat_idx:<3}] {feat_name:<25} {mae_change:>18.6f} {class_changes:>15}")
        else:
            diff = np.abs(pred_ablated - pred_baseline)
            mae_change = float(diff.mean())
            max_change = float(diff.max())

            ablation_results.append((feat_name, mae_change, max_change))
            print(f"  [{feat_idx:<3}] {feat_name:<25} {mae_change:>18.6f} {max_change:>15.6f}")

    # ── 6. Rangkuman Kontribusi terbesar ─────────────────────────────────────
    print(f"  {'-'*75}")
    ablation_results.sort(key=lambda x: x[1], reverse=True)

    if has_classifier:
        print(f"\n  Peringkat Kontribusi Kovariat terhadap {LABEL.upper()} (MAE Probabilitas terbesar):")
        for rank, (feat_name, mae_change, class_changes) in enumerate(ablation_results, 1):
            bar_len = int(mae_change / max(ablation_results[0][1], 1e-6) * 30)
            bar = "|" * bar_len
            print(f"  #{rank:<3} {feat_name:<25} {bar:<30} {mae_change:.6f} ({class_changes} jam)")
    else:
        print(f"\n  Peringkat Kontribusi Kovariat terhadap {LABEL.upper()} (MAE Perubahan terbesar):")
        for rank, (feat_name, mae_change, max_change) in enumerate(ablation_results, 1):
            bar_len = int(mae_change / max(ablation_results[0][1], 1e-6) * 30)
            bar = "|" * bar_len
            print(f"  #{rank:<3} {feat_name:<25} {bar:<30} {mae_change:.6f}")

    # ── 7. Kesimpulan ────────────────────────────────────────────────────────
    contributing = [r for r in ablation_results if r[1] > 1e-6]
    print(f"\n{'=' * 75}")
    print("  KESIMPULAN:")
    print(f"  Dari {len(ablation_results)} variabel kovariat yang diuji,")
    print(f"  sebanyak {len(contributing)} variabel terbukti memengaruhi prediksi {LABEL}")
    print(f"  (MAE Perubahan > 0) ketika dinolkan.")
    if len(contributing) > 0:
        print(f"\n  Hal ini membuktikan bahwa model {LABEL} tidak bersifat univariat")
        print("  murni, melainkan memanfaatkan kombinasi multivariabel di encoder")
        print("  serta classifier head untuk memproyeksikan cuaca.")
    print("=" * 75)


if __name__ == "__main__":
    main()

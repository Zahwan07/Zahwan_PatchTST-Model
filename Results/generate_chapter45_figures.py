"""
Generate tables and figures for thesis Bab 4.5 (Evaluation).

Run from project root:
  pip install matplotlib
  python Results/generate_chapter45_figures.py

Outputs: Results/chapter45_figures/
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Multi_Model.datasets_multi import load_historical_df
from Multi_Model.multi_config import (
    FEATURES_BY_LABEL,
    INPUT_LEN,
    PATCH_LEN,
    REGRESSION_LABELS,
    STRIDE,
)
# Exclude ph model as it is no longer used
REGRESSION_LABELS = [l for l in REGRESSION_LABELS if l != "ph"]

from Main_model.preprocessor import EnvironmentPreprocessor
from model.patchtst_official import PatchTST_Official

# Reuse eval helpers from Results/eval_artifacts.py
sys.path.insert(0, os.path.join(ROOT, "Results"))
import eval_artifacts as eva

OUT_DIR = os.path.join(ROOT, "Results", "chapter45_figures")
SEED = 42

# Manual Google comparison (3-hour slots, °C) — from thesis draft; "aktual" = referensi Google
GOOGLE_ROWS = [
    # date, time, prediksi_model, referensi_google
    # 23 Mei
    ("2026-05-23", "00:00", 21.3, 21.0),
    ("2026-05-23", "03:00", 20.1, 20.0),
    ("2026-05-23", "06:00", 20.0, 20.0),
    ("2026-05-23", "09:00", 25.4, 24.0),
    ("2026-05-23", "12:00", 27.7, 26.0),
    ("2026-05-23", "15:00", 26.3, 26.0),
    ("2026-05-23", "18:00", 23.4, 23.0),
    ("2026-05-23", "21:00", 22.0, 22.0),
    # 24 Mei
    ("2026-05-24", "00:00", 21.3, 20.0),
    ("2026-05-24", "03:00", 20.1, 19.0),
    ("2026-05-24", "06:00", 19.8, 19.0),
    ("2026-05-24", "09:00", 25.3, 24.0),
    ("2026-05-24", "12:00", 27.9, 26.0),
    ("2026-05-24", "15:00", 26.3, 26.0),
    ("2026-05-24", "18:00", 23.5, 23.0),
    ("2026-05-24", "21:00", 21.9, 21.0),
    # 25 Mei
    ("2026-05-25", "00:00", 21.1, 20.0),
    ("2026-05-25", "03:00", 19.8, 20.0),
    ("2026-05-25", "06:00", 19.8, 19.0),
    ("2026-05-25", "09:00", 25.3, 24.0),
    ("2026-05-25", "12:00", 27.8, 26.0),
    ("2026-05-25", "15:00", 26.5, 26.0),
    ("2026-05-25", "18:00", 23.5, 23.0),
    ("2026-05-25", "21:00", 22.1, 22.0),
    # 26 Mei
    ("2026-05-26", "00:00", 21.2, 20.0),
    ("2026-05-26", "03:00", 20.3, 19.0),
    ("2026-05-26", "06:00", 20.0, 19.0),
    ("2026-05-26", "09:00", 25.3, 24.0),
    ("2026-05-26", "12:00", 27.5, 27.0),
    ("2026-05-26", "15:00", 26.2, 27.0),
    ("2026-05-26", "18:00", 23.8, 23.0),
    ("2026-05-26", "21:00", 22.2, 22.0),
    # 27 Mei
    ("2026-05-27", "00:00", 21.1, 21.0),
    ("2026-05-27", "03:00", 20.0, 19.0),
    ("2026-05-27", "06:00", 20.0, 19.0),
    ("2026-05-27", "09:00", 25.5, 24.0),
    ("2026-05-27", "12:00", 27.5, 27.0),
    ("2026-05-27", "15:00", 26.5, 27.0),
    ("2026-05-27", "18:00", 23.9, 23.0),
    ("2026-05-27", "21:00", 22.3, 21.0),
    # 28 Mei
    ("2026-05-28", "00:00", 21.2, 20.0),
    ("2026-05-28", "03:00", 20.0, 19.0),
    ("2026-05-28", "06:00", 19.7, 19.0),
    ("2026-05-28", "09:00", 25.3, 25.0),
    ("2026-05-28", "12:00", 27.5, 28.0),
    ("2026-05-28", "15:00", 26.4, 27.0),
    ("2026-05-28", "18:00", 24.0, 24.0),
    ("2026-05-28", "21:00", 22.3, 22.0),
    # 29 Mei
    ("2026-05-29", "00:00", 21.2, 20.0),
    ("2026-05-29", "03:00", 20.3, 19.0),
    ("2026-05-29", "06:00", 20.0, 19.0),
    ("2026-05-29", "09:00", 25.3, 24.0),
    ("2026-05-29", "12:00", 27.5, 27.0),
    ("2026-05-29", "15:00", 26.2, 27.0),
    ("2026-05-29", "18:00", 23.8, 23.0),
    ("2026-05-29", "21:00", 22.2, 22.0),
    # 30 Mei
    ("2026-05-30", "00:00", 21.4, 20.0),
    ("2026-05-30", "03:00", 20.2, 19.0),
    ("2026-05-30", "06:00", 19.8, 20.0),
    ("2026-05-30", "09:00", 25.4, 24.0),
    ("2026-05-30", "12:00", 27.5, 27.0),
    ("2026-05-30", "15:00", 26.0, 26.0),
    ("2026-05-30", "18:00", 24.0, 23.0),
    ("2026-05-30", "21:00", 22.3, 21.0),
]

VARIANTS = [
    ("Konfigurasi utama (720 jam, 70/30)", "Main_model/Multi_Model/artifacts", 720, 0.7),
    ("720 jam, 70/30", "Results/720_70-30", 720, 0.7),
    ("720 jam, 80/20", "Results/720_80-20", 720, 0.8),
    ("336 jam, 70/30", "Results/336-70-30", 336, 0.7),
    ("336 jam, 80/20", "Results/336-80-20", 336, 0.8),
]


def _ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def run_all_metrics(df, device) -> pd.DataFrame:
    rows = []
    for name, art_dir, pred_len, train_frac in VARIANTS:
        abs_dir = os.path.join(ROOT, art_dir.replace("/", os.sep))
        if not os.path.isdir(abs_dir):
            print(f"Skip (missing dir): {name}")
            continue
        val_idx = eva._make_val_indices(len(df), INPUT_LEN, pred_len, train_frac, SEED)
        for label in REGRESSION_LABELS:
            try:
                mae, mse, rmse, mape, n_feat = eva._evaluate_regression_label(
                    label, df, val_idx, device, pred_len, art_dir
                )
                rows.append(
                    {
                        "varian": name,
                        "pred_len": pred_len,
                        "train_fraction": train_frac,
                        "label": label,
                        "MAE": round(mae, 4),
                        "MSE": round(mse, 6),
                        "RMSE": round(rmse, 4),
                        "MAPE_pct": round(mape, 2),
                        "n_features": n_feat,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "varian": name,
                        "pred_len": pred_len,
                        "train_fraction": train_frac,
                        "label": label,
                        "MAE": np.nan,
                        "MSE": np.nan,
                        "RMSE": np.nan,
                        "MAPE_pct": np.nan,
                        "n_features": np.nan,
                        "error": str(e),
                    }
                )
        try:
            acc, n_feat = eva._evaluate_cuaca(df, val_idx, pred_len, art_dir)
            rows.append(
                {
                    "varian": name,
                    "pred_len": pred_len,
                    "train_fraction": train_frac,
                    "label": "cuaca",
                    "MAE": np.nan,
                    "MSE": np.nan,
                    "RMSE": np.nan,
                    "MAPE_pct": np.nan,
                    "accuracy_pct": round(acc, 2),
                    "n_features": n_feat,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "varian": name,
                    "pred_len": pred_len,
                    "train_fraction": train_frac,
                    "label": "cuaca",
                    "accuracy_pct": np.nan,
                    "error": str(e),
                }
            )
    return pd.DataFrame(rows)


def plot_rmse_bars(metrics_df: pd.DataFrame, varian_filter: str, fname: str, title: str):
    sub = metrics_df[
        (metrics_df["varian"] == varian_filter)
        & (metrics_df["label"].isin(REGRESSION_LABELS))
        & metrics_df["RMSE"].notna()
    ]
    if sub.empty:
        return
    labels = sub["label"].tolist()
    vals = sub["RMSE"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, vals, color="#4C72B0")
    ax.set_ylabel("RMSE (skala terstandarisasi)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)


def plot_variant_comparison_suhu(metrics_df: pd.DataFrame):
    sub = metrics_df[
        (metrics_df["label"] == "suhu") & metrics_df["RMSE"].notna()
    ].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(sub))
    ax.bar(x, sub["RMSE"], color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['pred_len']}h\n{int(r['train_fraction']*100)}/{int((1-r['train_fraction'])*100)}"
         for _, r in sub.iterrows()],
        rotation=0,
        fontsize=9,
    )
    ax.set_ylabel("RMSE suhu (skala terstandarisasi)")
    ax.set_title("Perbandingan RMSE suhu antar variasi horizon dan split data")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grafik_3_rmse_suhu_varian.png"), dpi=150)
    plt.close(fig)


def plot_suhu_timeseries_sample(df, device, art_dir="Main_model/Multi_Model/artifacts", pred_len=720, train_frac=0.7):
    """Grafik 4: satu sampel validasi — 7 hari pertama prediksi suhu vs aktual (°C)."""
    val_idx = eva._make_val_indices(len(df), INPUT_LEN, pred_len, train_frac, SEED)
    if len(val_idx) == 0:
        return
    j = int(val_idx[len(val_idx) // 2])
    features = FEATURES_BY_LABEL["suhu"]
    pre = EnvironmentPreprocessor(features=features, inverse_output="subset")
    pre.fit(df)
    data_scaled = pre.transform(df)
    model = PatchTST_Official(
        input_dim=data_scaled.shape[-1],
        input_len=INPUT_LEN,
        pred_len=pred_len,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        d_model=128,
        n_heads=16,
        num_layers=3,
        dropout=0.2,
        revin=True,
    ).to(device)
    path = os.path.join(ROOT, art_dir, "patchtst_suhu.pth")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    x = torch.tensor(data_scaled[j : j + INPUT_LEN], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x).cpu().numpy().reshape(pred_len, -1)
    y_true = data_scaled[j + INPUT_LEN : j + INPUT_LEN + pred_len]
    pred_c = pre.inverse_transform(out)[:, 0]
    true_c = pre.inverse_transform(y_true)[:, 0]
    hours = min(24 * 7, pred_len)
    t = np.arange(hours)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, true_c[:hours], label="Aktual (validasi)", linewidth=1.5)
    ax.plot(t, pred_c[:hours], label="Prediksi model", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Jam ke-depan (dari akhir jendela masukan)")
    ax.set_ylabel("Suhu (°C)")
    ax.set_title("Contoh perbandingan prediksi vs aktual — suhu (7 hari pertama)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grafik_4_suhu_pred_vs_aktual_7hari.png"), dpi=150)
    plt.close(fig)


def cuaca_confusion_matrix(df, art_dir, pred_len, train_frac, device) -> pd.DataFrame:
    val_idx = eva._make_val_indices(len(df), INPUT_LEN, pred_len, train_frac, SEED)
    pre = joblib_load_cuaca_pre(art_dir)
    data_scaled = pre.transform(df)
    model = eva.CuacaPatchTST(input_dim=data_scaled.shape[-1], pred_len=pred_len)
    payload = torch.load(os.path.join(ROOT, art_dir, "patchtst_cuaca.pth"), map_location="cpu")
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    p = df["precipitation"].to_numpy(dtype=float)
    y_cuaca = np.where(p > 0.5, 2, np.where(p > 0.1, 1, 0)).astype(np.int64)
    y_true_all, y_pred_all = [], []
    for j in val_idx[: min(200, len(val_idx))]:
        xb = torch.tensor(data_scaled[j : j + INPUT_LEN], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(xb)
            pred = torch.argmax(logits, dim=-1).cpu().numpy().ravel()
        true = y_cuaca[j + INPUT_LEN : j + INPUT_LEN + pred_len]
        y_true_all.extend(true.tolist())
        y_pred_all.extend(pred.tolist())
    cm = np.zeros((3, 3), dtype=int)
    for t, p_ in zip(y_true_all, y_pred_all):
        if 0 <= t <= 2 and 0 <= p_ <= 2:
            cm[t, p_] += 1
    labels = ["Cerah (0)", "Berawan (1)", "Hujan (2)"]
    cm_df = pd.DataFrame(cm, index=[f"Aktual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Cerah", "Berawan", "Hujan"])
    ax.set_yticklabels(["Cerah", "Berawan", "Hujan"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Matriks kebingungan cuaca (sampel validasi)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grafik_5_confusion_cuaca.png"), dpi=150)
    plt.close(fig)
    return cm_df


def joblib_load_cuaca_pre(art_dir):
    import joblib
    return joblib.load(os.path.join(ROOT, art_dir, "preprocessor_cuaca.joblib"))


def build_google_table() -> pd.DataFrame:
    rows = []
    for date_s, time_s, pred, google in GOOGLE_ROWS:
        rows.append(
            {
                "tanggal": date_s,
                "waktu": time_s,
                "prediksi_model_C": pred,
                "referensi_google_C": google,
                "selisih_model_minus_google": pred - google if pd.notna(google) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_google_comparison(gdf: pd.DataFrame):
    sub = gdf.dropna(subset=["referensi_google_C"]).copy()
    if sub.empty:
        return
    
    # Format labels for line plot
    sub["label"] = sub["tanggal"].apply(lambda x: x.split("-")[-1] + " Mei") + "\n" + sub["waktu"]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(sub))
    
    # Beautiful line plot
    ax.plot(x, sub["prediksi_model_C"], label="Prediksi Model", color="#1F77B4", linewidth=2, marker='o', markersize=4)
    ax.plot(x, sub["referensi_google_C"], label="Referensi Google (Aktual)", color="#FF7F0E", linewidth=2, marker='s', markersize=4, linestyle='--')
    
    # Label every 4th point (12 hours) to avoid clutter, but show dates clearly
    tick_indices = np.arange(0, len(sub), 4)
    tick_labels = [sub.iloc[i]["label"] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    
    ax.set_ylabel("Suhu (°C)", fontsize=10)
    ax.set_title("Perbandingan Suhu — Model vs Google (23–30 Mei 2026, interval 3 jam)", fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grafik_1_suhu_model_vs_google.png"), dpi=150)
    plt.close(fig)

    mae_m = float(np.mean(np.abs(sub["selisih_model_minus_google"])))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["Model vs Google"], [mae_m], color=["#4C72B0"], width=0.4)
    ax.set_ylabel("MAE (°C)", fontsize=10)
    ax.set_title(f"Rata-rata Selisih Absolut\n(23–30 Mei, n={len(sub)})", fontsize=11, fontweight='bold')
    ax.text(0, mae_m + 0.01, f"{mae_m:.4f} °C", ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, mae_m + 0.15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grafik_2_mae_model_vs_google.png"), dpi=150)
    plt.close(fig)
    return mae_m


def plot_learning_curves():
    # ─── Learning curve data: MultivariatePatchTST Regression (720 jam, 80:20)
    # Data dari training log terbaru — MSE loss per epoch (seed=42)
    history = {
        "suhu": {
            # 300 epochs (best val @ epoch 280, val_loss=0.120441)
            "epoch": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                      100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                      200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 299],
            "train": [0.226752, 0.149843, 0.145725, 0.142655, 0.140697, 0.139118,
                      0.137838, 0.136527, 0.135856, 0.134819, 0.134099, 0.133611,
                      0.132886, 0.132130, 0.132124, 0.131490, 0.131153, 0.130712,
                      0.130517, 0.130321, 0.129956, 0.129491, 0.129235, 0.129161,
                      0.128587, 0.128645, 0.128144, 0.128027, 0.127540, 0.127271, 0.127205],
            "val":   [0.182063, 0.147980, 0.142436, 0.139725, 0.136990, 0.136632,
                      0.133311, 0.132671, 0.132694, 0.130401, 0.129328, 0.129624,
                      0.127440, 0.127462, 0.127368, 0.127123, 0.127339, 0.126144,
                      0.126559, 0.126918, 0.124009, 0.124308, 0.125431, 0.123354,
                      0.123657, 0.123735, 0.122215, 0.121556, 0.120441, 0.120738, 0.121321],
        },
        "humidity": {
            # 300 epochs (best val @ epoch 290, val_loss=0.153717)
            "epoch": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                      100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                      200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 299],
            "train": [0.374449, 0.243066, 0.229600, 0.220957, 0.214785, 0.209267,
                      0.205437, 0.201328, 0.198549, 0.196086, 0.193847, 0.191910,
                      0.190476, 0.188345, 0.186933, 0.185547, 0.184788, 0.183547,
                      0.182189, 0.181698, 0.180547, 0.179342, 0.178853, 0.178048,
                      0.177391, 0.176728, 0.176394, 0.176049, 0.175394, 0.175021, 0.174618],
            "val":   [0.305160, 0.239880, 0.223017, 0.212118, 0.204071, 0.195573,
                      0.191571, 0.185350, 0.182687, 0.178727, 0.178593, 0.176467,
                      0.173644, 0.169067, 0.169312, 0.167431, 0.164036, 0.164703,
                      0.162059, 0.162142, 0.161107, 0.160404, 0.159097, 0.157550,
                      0.156367, 0.156358, 0.155474, 0.155511, 0.155106, 0.153717, 0.154365],
        },
        "light_intensity": {
            # Early stopping @ epoch 150 (best val @ epoch 120, val_loss=0.100762)
            "epoch": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                      100, 110, 120, 130, 140, 150],
            "train": [0.169381, 0.103560, 0.103054, 0.102805, 0.102605, 0.102295,
                      0.102331, 0.102067, 0.102053, 0.101996, 0.102042, 0.101951,
                      0.101914, 0.101910, 0.101975, 0.101891],
            "val":   [0.118289, 0.102554, 0.104494, 0.101708, 0.101475, 0.101374,
                      0.101572, 0.101270, 0.101218, 0.101310, 0.101741, 0.100900,
                      0.100762, 0.101108, 0.100903, 0.100809],
        },
        "precipitation": {
            # 300 epochs (best val @ epoch 299, val_loss=0.749416)
            "epoch": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                      100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                      200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 299],
            "train": [1.026997, 0.905492, 0.898900, 0.891516, 0.885989, 0.881868,
                      0.877537, 0.874046, 0.871171, 0.868476, 0.864821, 0.861830,
                      0.858407, 0.855599, 0.852249, 0.849050, 0.845085, 0.841308,
                      0.837234, 0.833226, 0.829416, 0.826084, 0.822414, 0.819271,
                      0.815671, 0.812716, 0.809905, 0.806936, 0.804184, 0.801416, 0.799513],
            "val":   [0.931784, 0.896405, 0.890211, 0.881300, 0.873895, 0.868412,
                      0.862778, 0.859671, 0.854647, 0.851359, 0.847371, 0.844252,
                      0.838216, 0.834279, 0.829962, 0.825679, 0.819698, 0.814258,
                      0.807605, 0.802074, 0.795980, 0.793121, 0.787412, 0.780262,
                      0.776362, 0.773112, 0.767923, 0.762577, 0.757546, 0.759482, 0.749416],
        },
        "cuaca": {
            # 300 epochs — Cross-Entropy loss + Val Accuracy (classifier, 3 kelas)
            "epoch": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                      100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                      200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 299],
            "train": [1.007960, 0.945669, 0.937066, 0.928559, 0.920589, 0.913896,
                      0.907556, 0.902277, 0.897770, 0.893752, 0.890225, 0.886996,
                      0.883575, 0.880146, 0.876738, 0.873604, 0.870758, 0.868282,
                      0.866131, 0.863877, 0.861799, 0.860368, 0.858859, 0.857110,
                      0.855951, 0.854102, 0.852255, 0.851156, 0.849669, 0.848001, 0.846595],
            "val":   [0.968936, 0.944020, 0.935698, 0.925143, 0.914659, 0.904973,
                      0.899775, 0.891579, 0.884176, 0.878534, 0.876163, 0.869875,
                      0.862997, 0.861420, 0.856311, 0.850877, 0.847921, 0.843239,
                      0.852202, 0.838488, 0.845772, 0.833317, 0.831697, 0.829494,
                      0.829817, 0.825864, 0.828127, 0.827392, 0.820244, 0.815626, 0.815931],            
        },
    }

    # Best val loss marker per label
    best_epochs = {
        "suhu": 280,
        "humidity": 290,
        "light_intensity": 120,
        "precipitation": 299,
        "cuaca": 290,   # val_loss=0.815626, ACC=69.13%
    }

    labels_to_plot = ["suhu", "humidity", "light_intensity", "precipitation", "cuaca"]
    titles = {
        "suhu":            "Suhu (Temperature)",
        "humidity":        "Kelembapan (Humidity)",
        "light_intensity": "Intensitas Cahaya (Light)",
        "precipitation":   "Curah Hujan (Precipitation)",
        "cuaca":           "Cuaca (Weather Classification)",
    }
    colors_train = "#1F77B4"
    colors_val   = "#FF7F0E"
    color_acc    = "#2CA02C"  # hijau untuk accuracy line

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.ravel()

    for i, label in enumerate(labels_to_plot):
        ax = axes[i]
        data = history[label]
        epochs = data["epoch"]
        train_loss = data["train"]
        val_loss   = data["val"]

        ax.plot(epochs, train_loss, label="Train Loss",
                color=colors_train, marker='o', markersize=3, linewidth=1.8)
        ax.plot(epochs, val_loss, label="Val Loss",
                color=colors_val, marker='s', markersize=3, linewidth=1.8, linestyle="--")

        # Mark best val epoch with a vertical dashed line
        best_ep = best_epochs[label]
        if best_ep in epochs:
            idx = epochs.index(best_ep)
            best_val = val_loss[idx]
            ax.axvline(best_ep, color="green", linestyle=":", linewidth=1.2, alpha=0.8)
            ax.annotate(
                f"best\nep {best_ep}\n({best_val:.4f})",
                xy=(best_ep, best_val),
                xytext=(best_ep + max(epochs) * 0.05, best_val + (max(val_loss) - min(val_loss)) * 0.08),
                fontsize=7.5, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
            )

        ax.set_title(titles[label], fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.55)

        # Cuaca: tambah sumbu kanan untuk Val Accuracy (%)
        if label == "cuaca" in data:
            ax2 = ax.twinx()
            ax2.tick_params(axis='y', labelcolor=color_acc)
            ax2.set_ylim(60, 75)
            # Merge legends dari kedua axis
            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower left")
        else:
            ax.legend(fontsize=8.5)

    # Sembunyikan subplot ke-6 (tidak terpakai)
    axes[5].axis("off")

    plt.suptitle(
        "Learning Curve Model — 720 Jam, Data Split 80:20",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grafik_6_learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    _ensure_out()
    print(f"Output directory: {OUT_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_historical_df()

    print("Computing metrics (may take several minutes)...")
    metrics_df = run_all_metrics(df, device)
    metrics_path = os.path.join(OUT_DIR, "tabel_4_metrik_regresi_semua_varian.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    main_name = "Konfigurasi utama (720 jam, 70/30)"
    reg_main = metrics_df[
        (metrics_df["varian"] == main_name) & metrics_df["label"].isin(REGRESSION_LABELS)
    ][["label", "MAE", "RMSE", "MAPE_pct"]]
    t4 = os.path.join(OUT_DIR, "tabel_4_ringkasan_metrik_konfigurasi_utama.csv")
    reg_main.to_csv(t4, index=False)

    cuaca_main = metrics_df[(metrics_df["varian"] == main_name) & (metrics_df["label"] == "cuaca")]
    t6 = os.path.join(OUT_DIR, "tabel_6_akurasi_cuaca.csv")
    cuaca_main[["varian", "accuracy_pct", "n_features"]].to_csv(t6, index=False)

    suhu_var = metrics_df[metrics_df["label"] == "suhu"][
        ["varian", "pred_len", "train_fraction", "MAE", "RMSE", "MAPE_pct"]
    ]
    t5 = os.path.join(OUT_DIR, "tabel_5_perbandingan_suhu_varian.csv")
    suhu_var.to_csv(t5, index=False)

    plot_rmse_bars(
        metrics_df,
        main_name,
        "grafik_3_rmse_per_label_utama.png",
        "RMSE per parameter (konfigurasi utama, skala terstandarisasi)",
    )
    plot_variant_comparison_suhu(metrics_df)
    print("Saved bar charts (grafik 3).")

    try:
        plot_suhu_timeseries_sample(df, device)
        print("Saved grafik_4_suhu_pred_vs_aktual_7hari.png")
    except Exception as e:
        print(f"Skip grafik 4: {e}")

    try:
        cm_df = cuaca_confusion_matrix(df, "Main_model/Multi_Model/artifacts", 720, 0.7, device)
        cm_df.to_csv(os.path.join(OUT_DIR, "tabel_confusion_cuaca.csv"))
        print("Saved grafik_5_confusion_cuaca.png")
    except Exception as e:
        print(f"Skip confusion matrix: {e}")

    gdf = build_google_table()
    gdf.to_csv(os.path.join(OUT_DIR, "tabel_1_perbandingan_google.csv"), index=False)
    mae_m = plot_google_comparison(gdf)
    summary_google = pd.DataFrame(
        [{"metode": "Model vs Google (23–30 Mei)", "MAE_C": mae_m, "n_titik": len(gdf.dropna(subset=["referensi_google_C"]))}]
    )
    summary_google.to_csv(os.path.join(OUT_DIR, "tabel_ringkasan_mae_google.csv"), index=False)
    print("Saved Google tables and grafik 1–2.")

    try:
        plot_learning_curves()
        print("Saved grafik_6_learning_curves.png")
    except Exception as e:
        print(f"Skip learning curves: {e}")

  # README for thesis insertion
    readme = os.path.join(OUT_DIR, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "Bab 4.5 — pemetaan file ke naskah\n"
            "================================\n"
            "Tabel 1 / Grafik 1–2: tabel_1_perbandingan_google.csv, grafik_1_*, grafik_2_*\n"
            "Tabel 4: tabel_4_ringkasan_metrik_konfigurasi_utama.csv\n"
            "Tabel 5: tabel_5_perbandingan_suhu_varian.csv\n"
            "Tabel 6: tabel_6_akurasi_cuaca.csv\n"
            "Grafik 3: grafik_3_rmse_per_label_utama.png, grafik_3_rmse_suhu_varian.png\n"
            "Grafik 4: grafik_4_suhu_pred_vs_aktual_7hari.png\n"
            "Grafik 5: grafik_5_confusion_cuaca.png\n"
            "Grafik 6: grafik_6_learning_curves.png\n"
            "Lengkap: tabel_4_metrik_regresi_semua_varian.csv\n"
        )
    print(f"Done. See {readme}")


if __name__ == "__main__":
    main()

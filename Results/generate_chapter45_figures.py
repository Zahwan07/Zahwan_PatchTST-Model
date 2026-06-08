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
    ("2026-05-26", "00:00", 21.2, 20.0),
    ("2026-05-26", "03:00", 19.9, 19.0),
    ("2026-05-26", "06:00", 19.8, 20.0),
    ("2026-05-26", "09:00", 25.4, 26.0),
    ("2026-05-26", "12:00", 27.5, 28.0),
    ("2026-05-26", "15:00", 26.0, 27.0),
    ("2026-05-26", "18:00", 23.8, 22.0),
    ("2026-05-26", "21:00", 22.2, 22.0),
    ("2026-05-27", "00:00", 21.1, 21.0),
    ("2026-05-27", "03:00", 20.0, 19.0),
    ("2026-05-27", "06:00", 20.0, 19.0),
    ("2026-05-27", "09:00", 25.5, 24.0),
    ("2026-05-27", "12:00", 27.5, np.nan),
    ("2026-05-27", "15:00", 26.3, np.nan),
    ("2026-05-27", "18:00", 23.9, np.nan),
    ("2026-05-27", "21:00", 22.3, np.nan),
    ("2026-05-28", "00:00", 21.2, np.nan),
    ("2026-05-28", "03:00", 20.0, np.nan),
    ("2026-05-28", "06:00", 19.7, np.nan),
    ("2026-05-28", "09:00", 25.3, np.nan),
    ("2026-05-28", "12:00", 27.5, np.nan),
    ("2026-05-28", "15:00", 26.4, np.nan),
    ("2026-05-28", "18:00", 24.0, np.nan),
    ("2026-05-28", "21:00", 22.3, np.nan),
    ("2026-05-29", "00:00", 21.2, np.nan),
    ("2026-05-29", "03:00", 20.3, np.nan),
    ("2026-05-29", "06:00", 20.0, np.nan),
    ("2026-05-29", "09:00", 25.3, np.nan),
    ("2026-05-29", "12:00", 27.5, np.nan),
    ("2026-05-29", "15:00", 26.2, np.nan),
    ("2026-05-29", "18:00", 23.8, np.nan),
    ("2026-05-29", "21:00", 22.2, np.nan),
    ("2026-05-30", "00:00", 21.4, np.nan),
    ("2026-05-30", "03:00", 20.2, np.nan),
    ("2026-05-30", "06:00", 19.8, np.nan),
    ("2026-05-30", "09:00", 25.4, np.nan),
    ("2026-05-30", "12:00", 27.5, np.nan),
    ("2026-05-30", "15:00", 26.0, np.nan),
    ("2026-05-30", "18:00", 24.0, np.nan),
    ("2026-05-30", "21:00", 22.3, np.nan),
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
    sub["label"] = sub["tanggal"] + " " + sub["waktu"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(sub))
    w = 0.35
    ax.bar(x - w / 2, sub["prediksi_model_C"], width=w, label="Prediksi model")
    ax.bar(x + w / 2, sub["referensi_google_C"], width=w, label="Referensi Google")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["waktu"], rotation=0)
    ax.set_ylabel("Suhu (°C)")
    ax.set_title("Perbandingan suhu — model vs Google (26 Mei 2026, interval 3 jam)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grafik_1_suhu_model_vs_google.png"), dpi=150)
    plt.close(fig)

    mae_m = float(np.mean(np.abs(sub["selisih_model_minus_google"])))
    mae_g = 0.0  # placeholder if only model vs google
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Model vs Google"], [mae_m], color=["#4C72B0"])
    ax.set_ylabel("MAE (°C)")
    ax.set_title(f"Rata-rata selisih absolut\n(26 Mei, n={len(sub)})")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grafik_2_mae_model_vs_google.png"), dpi=150)
    plt.close(fig)
    return mae_m


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
        [{"metode": "Model vs Google (26 Mei)", "MAE_C": mae_m, "n_titik": len(gdf.dropna(subset=["referensi_google_C"]))}]
    )
    summary_google.to_csv(os.path.join(OUT_DIR, "tabel_ringkasan_mae_google.csv"), index=False)
    print("Saved Google tables and grafik 1–2.")

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
            "Lengkap: tabel_4_metrik_regresi_semua_varian.csv\n"
        )
    print(f"Done. See {readme}")


if __name__ == "__main__":
    main()

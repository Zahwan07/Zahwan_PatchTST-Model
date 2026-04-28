"""
Predict with 6 separate models (Multi_Model) and write combined forecast CSV:
  Main_model/prediction_forecast.csv

Outputs columns (same as Main_model.config_exp.FEATURES_FORECAST):
  suhu, cuaca, humidity, light_intensity, ph, precipitation

Train regression: python Multi_Model/train_multi.py
Train cuaca: python Multi_Model/train_cuaca.py  (needs cuaca artifacts below)

Run from project root:
  python Multi_Model/predict_multi.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.patchtst_official import PatchTST_Official
from Multi_Model.multi_config import (
    INPUT_LEN,
    PRED_LEN,
    PATCH_LEN,
    STRIDE,
    REGRESSION_LABELS,
    artifacts_for_label,
    FEATURES_BY_LABEL,
)
from Multi_Model.cuaca_classifier import predict_cuaca


DATA_PATH = os.path.join(ROOT, "data", "historical_environment.csv")
OUT_DIR = os.path.join(ROOT, "Main_model")
OUT_CSV = os.path.join(OUT_DIR, "prediction_forecast.csv")


def _load_patchtst(label: str, input_dim: int) -> PatchTST_Official:
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
    art = artifacts_for_label(ROOT, label)
    model.load_state_dict(torch.load(art.model_path, map_location="cpu"))
    model.eval()
    return model


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("time").reset_index(drop=True)

    # Build timestamps for forecast: first prediction is +1 hour after last row
    last_time = pd.to_datetime(df["time"].iloc[-1])
    pred_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=PRED_LEN, freq="h")

    # --- Regression models ---
    preds = {}
    for label in REGRESSION_LABELS:
        art = artifacts_for_label(ROOT, label)
        pre = joblib.load(art.preprocessor_path)
        feats = FEATURES_BY_LABEL[label]

        # last 168h window for this label's feature subset
        window_df = df.tail(INPUT_LEN)
        X_scaled = pre.transform(window_df)  # (168, D)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

        model = _load_patchtst(label, input_dim=X_scaled.shape[-1])
        with torch.no_grad():
            y_scaled = model(X_tensor).numpy().reshape(PRED_LEN, -1)

        # inverse -> subset columns; target is always index 0
        y_subset = pre.inverse_transform(y_scaled).reshape(PRED_LEN, -1)
        preds[label] = y_subset[:, 0].astype(float)

    # --- Cuaca classifier (multi-step; sklearn — not PatchTST) ---
    art_c = artifacts_for_label(ROOT, "cuaca")
    if not os.path.isfile(art_c.preprocessor_path) or not os.path.isfile(art_c.model_path):
        missing = []
        if not os.path.isfile(art_c.preprocessor_path):
            missing.append(art_c.preprocessor_path)
        if not os.path.isfile(art_c.model_path):
            missing.append(art_c.model_path)
        raise FileNotFoundError(
            "Cuaca artifacts missing:\n  "
            + "\n  ".join(missing)
            + "\nRun: python Multi_Model/train_cuaca.py"
        )
    pre_c = joblib.load(art_c.preprocessor_path)
    clf = joblib.load(art_c.model_path)
    window_df = df.tail(INPUT_LEN)
    Xc_scaled = pre_c.transform(window_df)  # (168, D)
    cuaca_pred = predict_cuaca(clf, Xc_scaled)  # (720,)

    # Assemble output
    out = pd.DataFrame(
        {
            "time": pred_times,
            "suhu": preds["suhu"],
            "cuaca": cuaca_pred.astype(float),
            "humidity": preds["humidity"],
            "light_intensity": preds["light_intensity"],
            "ph": preds["ph"],
            "precipitation": preds["precipitation"],
        }
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved combined forecast to: {OUT_CSV}")
    print(out.head(3).to_string(index=False))
    print("...")
    print(out.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()


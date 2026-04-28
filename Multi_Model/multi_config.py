"""
Multi-model configuration (6 models total at inference):
- 5 PatchTST regression models (train_multi.py): suhu, humidity, light_intensity, ph, precipitation
- 1 cuaca classifier sklearn (train_cuaca.py): 3 classes (clear/cloudy/rain), multi-step (PRED_LEN steps)

This module centralizes:
- feature subset per label (FEATURES_BY_LABEL)
- artifact paths per label (ARTIFACTS_DIR + helpers)
- default hyperparams (reuse Main_model.config_exp defaults)

NOTE: For each regression label, the target column MUST be the first feature in the list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from Main_model.config_exp import INPUT_LEN as DEFAULT_INPUT_LEN, PRED_LEN as DEFAULT_PRED_LEN


# -------------------------
# Defaults (initial baseline)
# -------------------------
INPUT_LEN = DEFAULT_INPUT_LEN   # 168
PRED_LEN = DEFAULT_PRED_LEN     # 720
PATCH_LEN = 16
STRIDE = 8


REGRESSION_LABELS = ["suhu", "humidity", "light_intensity", "ph", "precipitation"]
CLASSIFICATION_LABELS = ["cuaca"]


# -------------------------
# Feature subsets per label
# -------------------------
# Important: the first item must be the target label for regression tasks.
FEATURES_BY_LABEL: dict[str, list[str]] = {
    # Temperature: strong diurnal + radiation + cloud + lags + seasonal context
    "suhu": [
        "suhu",
        "light_intensity",
        "cloud_cover",
        "humidity",
        "dewpoint_2m",
        "wind_speed_10m",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "temp_lag_24",
        "temp_lag_168",
    ],
    # Humidity: depends on dewpoint/suhu/diurnal cycle + wind/cloud context
    "humidity": [
        "humidity",
        "suhu",
        "dewpoint_2m",
        "cloud_cover",
        "wind_speed_10m",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ],
    # Light intensity: driven by time-of-day/season + cloud cover
    "light_intensity": [
        "light_intensity",
        "cloud_cover",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ],
    # pH: slow drift in this project; keep small, stable context
    "ph": [
        "ph",
        "doy_sin",
        "doy_cos",
        "hour_sin",
        "hour_cos",
        "suhu",
        "humidity",
    ],
    # Precipitation: event-driven; use atmospheric context + time
    "precipitation": [
        "precipitation",
        "cloud_cover",
        "surface_pressure",
        "humidity",
        "dewpoint_2m",
        "wind_speed_10m",
        "wind_dir_sin",
        "wind_dir_cos",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ],
    # Cuaca classifier (multi-step): features excluding target
    "cuaca": [
        # no cuaca itself in inputs; derived target from precipitation in dataset builder
        "weather_code",
        "precipitation",
        "cloud_cover",
        "surface_pressure",
        "humidity",
        "dewpoint_2m",
        "wind_speed_10m",
        "wind_dir_sin",
        "wind_dir_cos",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "temp_lag_24",
        "temp_lag_168",
        "suhu",
        "light_intensity",
    ],
}


@dataclass(frozen=True)
class Artifacts:
    label: str
    model_path: str
    preprocessor_path: str | None = None


def artifacts_dir(project_root: str) -> str:
    return os.path.join(project_root, "Main_model", "Multi_Model", "artifacts")


def artifacts_for_label(project_root: str, label: str) -> Artifacts:
    out = artifacts_dir(project_root)
    if label in REGRESSION_LABELS:
        return Artifacts(
            label=label,
            model_path=os.path.join(out, f"patchtst_{label}.pth"),
            preprocessor_path=os.path.join(out, f"preprocessor_{label}.joblib"),
        )
    if label == "cuaca":
        return Artifacts(
            label=label,
            model_path=os.path.join(out, "cuaca_clf.joblib"),
            preprocessor_path=os.path.join(out, "preprocessor_cuaca.joblib"),
        )
    raise KeyError(f"Unknown label: {label}")


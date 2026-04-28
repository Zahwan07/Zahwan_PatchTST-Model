"""
Dataset helpers for Multi_Model (6 separate models).

Regression labels:
  - build X from selected feature subset (scaled)
  - build Y as the SAME feature subset (scaled), so PatchTST output_dim == input_dim
  - loss will be focused to the target feature (index 0) via channel weights in training

Cuaca classifier:
  - build X from selected feature subset (scaled)
  - build Y as cuaca classes (0/1/2) for the next pred_len steps (multi-output)
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

from Main_model.preprocessor import EnvironmentPreprocessor
from Multi_Model.multi_config import FEATURES_BY_LABEL, INPUT_LEN, PRED_LEN, REGRESSION_LABELS


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "historical_environment.csv")


def load_historical_df(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        # Keep time for optional downstream alignment; modeling uses numeric cols only.
        df = df.sort_values("time").reset_index(drop=True)
    return df


def precip_to_cuaca(p: np.ndarray) -> np.ndarray:
    """Same thresholds as fetch_openmeteo_historical.py."""
    p = np.asarray(p, dtype=float)
    return np.where(p > 0.5, 2, np.where(p > 0.1, 1, 0)).astype(np.int64)


def create_windows_xy(data: np.ndarray, input_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generic sliding window builder: X=(N,input_len,D), Y=(N,pred_len,D)."""
    X, Y = [], []
    for i in range(len(data) - input_len - pred_len):
        X.append(data[i : i + input_len])
        Y.append(data[i + input_len : i + input_len + pred_len])
    return np.asarray(X), np.asarray(Y)


def build_regression_dataset(
    label: str,
    df: pd.DataFrame | None = None,
    input_len: int = INPUT_LEN,
    pred_len: int = PRED_LEN,
) -> tuple[np.ndarray, np.ndarray, EnvironmentPreprocessor, list[str]]:
    """
    Returns:
      X: (N, input_len, D)
      Y: (N, pred_len, D)  -- same D as input for PatchTST
      preprocessor: fitted on subset features
      features: list[str] used (target must be features[0])
    """
    if label not in REGRESSION_LABELS:
        raise ValueError(f"label must be one of {REGRESSION_LABELS}, got {label}")
    if df is None:
        df = load_historical_df()

    features = FEATURES_BY_LABEL[label]
    if not features or features[0] != label:
        raise ValueError(f"FEATURES_BY_LABEL['{label}'] must start with target '{label}'.")

    # Fit preprocessor on subset; inverse_output='subset' so inference returns subset columns only
    pre = EnvironmentPreprocessor(features=features, inverse_output="subset")
    pre.fit(df)
    data_scaled = pre.transform(df)  # (T, D)

    X, Y = create_windows_xy(data_scaled, input_len=input_len, pred_len=pred_len)
    return X, Y, pre, features


def build_cuaca_dataset(
    df: pd.DataFrame | None = None,
    input_len: int = INPUT_LEN,
    pred_len: int = PRED_LEN,
) -> tuple[np.ndarray, np.ndarray, EnvironmentPreprocessor, list[str]]:
    """
    Multi-output classification dataset.

    Returns:
      X: (N, input_len, D)  -- scaled
      Y: (N, pred_len)      -- cuaca int per step
      preprocessor: fitted on subset features (scaled)
      features: list[str] used for X
    """
    if df is None:
        df = load_historical_df()

    features = FEATURES_BY_LABEL["cuaca"]
    pre = EnvironmentPreprocessor(features=features, inverse_output="subset")
    pre.fit(df)
    X_scaled = pre.transform(df)  # (T, D)

    # Build cuaca target from precipitation in raw df
    if "precipitation" not in df.columns:
        raise ValueError("historical_environment.csv must contain 'precipitation' to derive cuaca.")
    cuaca_series = precip_to_cuaca(df["precipitation"].to_numpy())

    X, _ = create_windows_xy(X_scaled, input_len=input_len, pred_len=pred_len)

    Y = []
    for i in range(len(cuaca_series) - input_len - pred_len):
        Y.append(cuaca_series[i + input_len : i + input_len + pred_len])
    Y = np.asarray(Y, dtype=np.int64)

    return X, Y, pre, features


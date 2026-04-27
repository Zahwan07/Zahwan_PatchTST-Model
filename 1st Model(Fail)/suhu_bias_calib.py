"""
Hourly suhu bias correction from validation residuals (true - pred) in °C, binned by local hour.

Saved as Main_model/suhu_hourly_bias_c.npy (shape (24,)). Applied at inference in predict_exp / serve_realtime.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

BIAS_FILENAME = "suhu_hourly_bias_c.npy"


def compute_suhu_hourly_bias_c(
    val_out: torch.Tensor,
    y_val: torch.Tensor,
    val_indices: torch.Tensor,
    times: pd.Series,
    input_len: int,
    pred_len: int,
    preprocessor,
) -> np.ndarray:
    """
    val_out, y_val: (N, pred_len, INPUT_DIM) scaled. val_indices: (N,) original window start row index.
    times: df['time'] aligned with scaled data rows (same length as raw CSV).
    """
    n = val_out.shape[0]
    vo = val_out.reshape(-1, val_out.shape[-1]).detach().cpu().numpy()
    yo = y_val.reshape(-1, y_val.shape[-1]).detach().cpu().numpy()
    vo_raw = preprocessor.scaler_.inverse_transform(vo)
    yo_raw = preprocessor.scaler_.inverse_transform(yo)
    suhu_i = preprocessor.continuous_cols_.index("suhu")
    pred_suhu = vo_raw[:, suhu_i]
    true_suhu = yo_raw[:, suhu_i]
    residual = true_suhu - pred_suhu

    hours = np.empty(n * pred_len, dtype=np.int64)
    for k in range(n):
        orig_i = int(val_indices[k].item())
        rows = orig_i + input_len + np.arange(pred_len, dtype=np.int64)
        tsub = pd.to_datetime(times.iloc[rows])
        hours[k * pred_len : (k + 1) * pred_len] = tsub.dt.hour.values

    sums = np.zeros(24, dtype=np.float64)
    counts = np.zeros(24, dtype=np.float64)
    for i, h in enumerate(hours):
        h = int(h) % 24
        sums[h] += residual[i]
        counts[h] += 1.0
    bias = np.zeros(24, dtype=np.float64)
    np.divide(sums, np.maximum(counts, 1.0), out=bias, where=counts > 0)
    return bias


def save_suhu_hourly_bias_c(bias: np.ndarray, out_dir: str) -> str:
    if bias.shape != (24,):
        raise ValueError("bias must have shape (24,)")
    path = os.path.join(out_dir, BIAS_FILENAME)
    np.save(path, bias.astype(np.float64))
    return path


def load_suhu_hourly_bias_c(out_dir: str) -> Optional[np.ndarray]:
    path = os.path.join(out_dir, BIAS_FILENAME)
    if not os.path.isfile(path):
        return None
    b = np.load(path)
    if b.shape != (24,):
        return None
    return b.astype(np.float64)

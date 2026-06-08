"""
Memory-safe evaluation for Multi_Model artifacts only.

Evaluates:
  - 5 PatchTST regression models: MAE/MSE/RMSE/MAPE (scaled, target channel)
  - 1 cuaca classifier: mean per-step accuracy

Run from project root:
  python Main_model/eval_exp.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import joblib

from model.patchtst_official import PatchTST_Official
from Multi_Model.datasets_multi import load_historical_df
from Multi_Model.multi_config import (
    FEATURES_BY_LABEL,
    INPUT_LEN,
    PATCH_LEN,
    PRED_LEN,
    REGRESSION_LABELS,
    STRIDE,
    artifacts_dir,
    artifacts_for_label,
)
from Main_model.preprocessor import EnvironmentPreprocessor

SEED = 42
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class CuacaPatchTST(torch.nn.Module):
    """Local classifier wrapper to evaluate cuaca checkpoint from train_multi.py."""

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
        self.classifier = torch.nn.Linear(input_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)


def _make_val_indices(total_rows: int, input_len: int, pred_len: int, seed: int = SEED) -> np.ndarray:
    n_samples = total_rows - input_len - pred_len
    if n_samples <= 0:
        raise ValueError("Not enough rows to build validation windows.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_end = int(0.8 * n_samples)
    return perm[train_end:]


def _iter_xy_batches(data_scaled: np.ndarray, indices: np.ndarray, input_len: int, pred_len: int, batch_size: int):
    for i in range(0, len(indices), batch_size):
        bi = indices[i : i + batch_size]
        xb = np.stack([data_scaled[j : j + input_len] for j in bi], axis=0)
        yb = np.stack([data_scaled[j + input_len : j + input_len + pred_len] for j in bi], axis=0)
        yield xb, yb


def _evaluate_regression_label(label: str, df, val_indices: np.ndarray, device: torch.device):
    features = FEATURES_BY_LABEL[label]
    pre = EnvironmentPreprocessor(features=features, inverse_output="subset")
    pre.fit(df)
    data_scaled = pre.transform(df)

    art = artifacts_for_label(ROOT, label)
    if not os.path.isfile(art.model_path) or not os.path.isfile(art.preprocessor_path):
        raise FileNotFoundError(f"Missing artifacts for '{label}'.")

    model = PatchTST_Official(
        input_dim=data_scaled.shape[-1],
        input_len=INPUT_LEN,
        pred_len=PRED_LEN,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        d_model=128,
        n_heads=16,
        num_layers=3,
        dropout=0.2,
        revin=True,
    ).to(device)
    model.load_state_dict(torch.load(art.model_path, map_location=device))
    model.eval()

    abs_sum = 0.0
    sq_sum = 0.0
    ape_sum = 0.0
    n_elem = 0
    eps = 1e-6

    with torch.no_grad():
        for xb_np, yb_np in _iter_xy_batches(
            data_scaled=data_scaled,
            indices=val_indices,
            input_len=INPUT_LEN,
            pred_len=PRED_LEN,
            batch_size=BATCH_SIZE,
        ):
            xb = torch.tensor(xb_np, dtype=torch.float32, device=device)
            yb = torch.tensor(yb_np, dtype=torch.float32, device=device)
            out = model(xb)[:, :, 0]      # target channel only
            tgt = yb[:, :, 0]
            err = out - tgt

            abs_sum += float(torch.abs(err).sum().item())
            sq_sum += float((err ** 2).sum().item())
            ape_sum += float((torch.abs(err) / torch.clamp(torch.abs(tgt), min=eps)).sum().item())
            n_elem += err.numel()

    mae = abs_sum / n_elem
    mse = sq_sum / n_elem
    rmse = float(np.sqrt(mse))
    mape = 100.0 * (ape_sum / n_elem)
    return mae, mse, rmse, mape, len(features)


def _evaluate_cuaca(df, val_indices: np.ndarray):
    art = artifacts_for_label(ROOT, "cuaca")
    if not os.path.isfile(art.model_path) or not os.path.isfile(art.preprocessor_path):
        raise FileNotFoundError("Missing cuaca artifacts.")

    features = FEATURES_BY_LABEL["cuaca"]
    pre = joblib.load(art.preprocessor_path)
    data_scaled = pre.transform(df)
    model = CuacaPatchTST(input_dim=data_scaled.shape[-1])
    payload = torch.load(art.model_path, map_location="cpu")
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    if "precipitation" not in df.columns:
        raise ValueError("Dataset must contain 'precipitation' to derive cuaca labels.")
    p = df["precipitation"].to_numpy(dtype=float)
    y_cuaca = np.where(p > 0.5, 2, np.where(p > 0.1, 1, 0)).astype(np.int64)

    correct = 0
    total = 0
    cls_batch_size = max(16, BATCH_SIZE // 2)
    for i in range(0, len(val_indices), cls_batch_size):
        start_indices = val_indices[i : i + cls_batch_size]
        xb_np = np.stack([data_scaled[j : j + INPUT_LEN] for j in start_indices], axis=0)
        y_true = np.stack(
            [y_cuaca[j + INPUT_LEN : j + INPUT_LEN + PRED_LEN] for j in start_indices],
            axis=0,
        )
        with torch.no_grad():
            logits = model(torch.tensor(xb_np, dtype=torch.float32))
            y_hat = torch.argmax(logits, dim=-1).cpu().numpy()
        correct += int((y_hat == y_true).sum())
        total += int(y_true.size)

    acc = (correct / total) * 100.0 if total > 0 else float("nan")
    return acc, len(features)


def main():
    multi_dir = artifacts_dir(ROOT)
    if not os.path.isdir(multi_dir):
        print(f"Multi-model artifacts directory not found: {multi_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("\n" + "=" * 72)
    print("--- Evaluation (Multi_Model only: 5 regression + cuaca classifier) ---")
    print(f"Artifacts dir: {multi_dir}")
    print("=" * 72)

    df = load_historical_df()
    val_indices = _make_val_indices(len(df), INPUT_LEN, PRED_LEN, seed=SEED)
    print(f"Validation samples: {len(val_indices)} (80/20 split, seed={SEED})")

    for label in REGRESSION_LABELS:
        try:
            mae, mse, rmse, mape, n_feats = _evaluate_regression_label(label, df=df, val_indices=val_indices, device=device)
            print(f"[{label}] scaled: MAE={mae:.4f} MSE={mse:.6f} RMSE={rmse:.4f} MAPE={mape:.2f}% | feats={n_feats}")
        except Exception as e:
            print(f"[{label}] evaluation skipped: {e}")

    try:
        acc, n_feats = _evaluate_cuaca(df=df, val_indices=val_indices)
        print(f"[cuaca] mean per-step accuracy: {acc:.2f}% | feats={n_feats}")
    except Exception as e:
        print(f"[cuaca] evaluation skipped: {e}")


if __name__ == "__main__":
    main()

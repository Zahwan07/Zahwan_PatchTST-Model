"""
Evaluate Multi_Model-style checkpoints from an arbitrary artifacts directory.

Same logic as Main_model/eval_exp.py:
  - Regression: refit EnvironmentPreprocessor on full historical df, scaled metrics
    on target channel (index 0), MAE/MSE/RMSE/MAPE on scaled tensors.
  - Cuaca: load saved preprocessor from disk, mean per-step accuracy.

Differences from eval_exp.py only:
  - Artifacts path and PRED_LEN / train split are CLI arguments (for results/* variants).

Run from project root (examples):
  python results/eval_artifacts.py --artifacts-dir results/720_80-20 --pred-len 720 --train-fraction 0.8
  python results/eval_artifacts.py --artifacts-dir results/336-70-30 --pred-len 336 --train-fraction 0.7
"""
from __future__ import annotations

import argparse
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
    REGRESSION_LABELS,
    STRIDE,
)
from Main_model.preprocessor import EnvironmentPreprocessor

SEED = 42
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


def _artifact_paths(artifacts_dir: str, label: str) -> tuple[str, str]:
    d = os.path.normpath(os.path.join(ROOT, artifacts_dir) if not os.path.isabs(artifacts_dir) else artifacts_dir)
    if label in REGRESSION_LABELS:
        return (
            os.path.join(d, f"patchtst_{label}.pth"),
            os.path.join(d, f"preprocessor_{label}.joblib"),
        )
    if label == "cuaca":
        return (
            os.path.join(d, "patchtst_cuaca.pth"),
            os.path.join(d, "preprocessor_cuaca.joblib"),
        )
    raise KeyError(label)


class CuacaPatchTST(torch.nn.Module):
    def __init__(self, input_dim: int, pred_len: int):
        super().__init__()
        self.backbone = PatchTST_Official(
            input_dim=input_dim,
            input_len=INPUT_LEN,
            pred_len=pred_len,
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


def _make_val_indices(
    total_rows: int,
    input_len: int,
    pred_len: int,
    train_fraction: float,
    seed: int = SEED,
) -> np.ndarray:
    n_samples = total_rows - input_len - pred_len
    if n_samples <= 0:
        raise ValueError("Not enough rows to build validation windows.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1 (exclusive).")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_end = int(train_fraction * n_samples)
    return perm[train_end:]


def _iter_xy_batches(
    data_scaled: np.ndarray,
    indices: np.ndarray,
    input_len: int,
    pred_len: int,
    batch_size: int,
):
    for i in range(0, len(indices), batch_size):
        bi = indices[i : i + batch_size]
        xb = np.stack([data_scaled[j : j + input_len] for j in bi], axis=0)
        yb = np.stack([data_scaled[j + input_len : j + input_len + pred_len] for j in bi], axis=0)
        yield xb, yb


def _evaluate_regression_label(
    label: str,
    df,
    val_indices: np.ndarray,
    device: torch.device,
    pred_len: int,
    artifacts_dir: str,
):
    features = FEATURES_BY_LABEL[label]
    pre = EnvironmentPreprocessor(features=features, inverse_output="subset")
    pre.fit(df)
    data_scaled = pre.transform(df)

    model_path, _ = _artifact_paths(artifacts_dir, label)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model for '{label}': {model_path}")

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
    model.load_state_dict(torch.load(model_path, map_location=device))
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
            pred_len=pred_len,
            batch_size=BATCH_SIZE,
        ):
            xb = torch.tensor(xb_np, dtype=torch.float32, device=device)
            yb = torch.tensor(yb_np, dtype=torch.float32, device=device)
            out = model(xb)[:, :, 0]
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


def _evaluate_cuaca(df, val_indices: np.ndarray, pred_len: int, artifacts_dir: str):
    model_path, pre_path = _artifact_paths(artifacts_dir, "cuaca")
    if not os.path.isfile(model_path) or not os.path.isfile(pre_path):
        raise FileNotFoundError("Missing cuaca artifacts.")

    features = FEATURES_BY_LABEL["cuaca"]
    pre = joblib.load(pre_path)
    data_scaled = pre.transform(df)
    model = CuacaPatchTST(input_dim=data_scaled.shape[-1], pred_len=pred_len)
    payload = torch.load(model_path, map_location="cpu")
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
            [y_cuaca[j + INPUT_LEN : j + INPUT_LEN + pred_len] for j in start_indices],
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
    p = argparse.ArgumentParser(description="Eval Multi_Model checkpoints (same metrics as eval_exp.py).")
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default=os.path.join("Main_model", "Multi_Model", "artifacts"),
        help="Folder containing patchtst_*.pth (+ preprocessor_cuaca.joblib for cuaca).",
    )
    p.add_argument(
        "--pred-len",
        type=int,
        required=True,
        help="Forecast horizon used when training (must match checkpoint head size).",
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of shuffled window indices used for TRAINING; validation is the remainder (e.g. 0.8 => 80/20, 0.7 => 70/30).",
    )
    p.add_argument("--seed", type=int, default=SEED, help="NumPy RNG seed for val index permutation.")
    args = p.parse_args()

    artifacts_dir = args.artifacts_dir
    pred_len = args.pred_len
    train_fraction = args.train_fraction
    seed = args.seed

    abs_ad = os.path.join(ROOT, artifacts_dir) if not os.path.isabs(artifacts_dir) else artifacts_dir
    if not os.path.isdir(abs_ad):
        print(f"Artifacts directory not found: {abs_ad}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("\n" + "=" * 72)
    print("--- Evaluation (Multi_Model: 5 regression + cuaca) ---")
    print(f"Artifacts dir: {abs_ad}")
    print(f"PRED_LEN={pred_len} | train_fraction={train_fraction} (val ~ {100 * (1 - train_fraction):.0f}%) | seed={seed}")
    print("=" * 72)

    df = load_historical_df()
    val_indices = _make_val_indices(len(df), INPUT_LEN, pred_len, train_fraction, seed=seed)
    val_pct = 100.0 * (1.0 - train_fraction)
    print(f"Validation samples: {len(val_indices)} (~{val_pct:.0f}% of windows, seed={seed})")

    for label in REGRESSION_LABELS:
        try:
            mae, mse, rmse, mape, n_feats = _evaluate_regression_label(
                label, df=df, val_indices=val_indices, device=device, pred_len=pred_len, artifacts_dir=artifacts_dir
            )
            print(f"[{label}] scaled: MAE={mae:.4f} MSE={mse:.6f} RMSE={rmse:.4f} MAPE={mape:.2f}% | feats={n_feats}")
        except Exception as e:
            print(f"[{label}] evaluation skipped: {e}")

    try:
        acc, n_feats = _evaluate_cuaca(df=df, val_indices=val_indices, pred_len=pred_len, artifacts_dir=artifacts_dir)
        print(f"[cuaca] mean per-step accuracy: {acc:.2f}% | feats={n_feats}")
    except Exception as e:
        print(f"[cuaca] evaluation skipped: {e}")


if __name__ == "__main__":
    main()

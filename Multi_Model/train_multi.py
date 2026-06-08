"""
Train 6 models in one script (Multi_Model):
  - 5 PatchTST regression models: suhu, humidity, light_intensity, ph, precipitation
  - 1 PatchTST cuaca classifier (3 classes: 0 clear, 1 cloudy, 2 rain)

Artifacts are saved to: Main_model/Multi_Model/artifacts/

Run from project root:
  python Multi_Model/train_multi.py
  python Multi_Model/train_multi.py --cuaca-only   # only refresh cuaca classifier artifacts
"""

from __future__ import annotations

import argparse
import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.patchtst_official import PatchTST_Official
from utils import compute_metrics
from Multi_Model.datasets_multi import load_historical_df, build_regression_dataset, build_cuaca_dataset
from Multi_Model.multi_config import (
    INPUT_LEN,
    PRED_LEN,
    PATCH_LEN,
    STRIDE,
    REGRESSION_LABELS,
    artifacts_for_label,
    artifacts_dir,
)


SEED = 42
EPOCHS = 300
PATIENCE = 25


def _set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_val_indices(n: int, train_fraction: float = 0.8, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle window indices once; avoid materializing full X/Y torch tensors on CPU."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_end = int(train_fraction * n)
    return perm[:train_end], perm[train_end:]


class _IndexedWindowDataset(Dataset):
    """Lazy batches from numpy windows — one copy in RAM, no torch.tensor(N, ...) duplicate."""

    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray, y_dtype: torch.dtype = torch.float32):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = y
        self.indices = indices
        self.y_dtype = y_dtype

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        xb = torch.from_numpy(self.x[j])
        yj = self.y[j]
        if self.y_dtype == torch.long:
            yb = torch.tensor(yj, dtype=torch.long)
        else:
            yb = torch.from_numpy(np.ascontiguousarray(yj, dtype=np.float32))
        return xb, yb


class CuacaPatchTST(torch.nn.Module):
    """PatchTST backbone + 3-class head for multi-step cuaca classification."""

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
        feat = self.backbone(x)       # (B, pred_len, input_dim)
        return self.classifier(feat)  # (B, pred_len, 3)


def _train_patchtst_single_label(
    label: str,
    df,
    device: torch.device,
):
    # Build dataset for label
    X, Y, preprocessor, features = build_regression_dataset(label, df=df, input_len=INPUT_LEN, pred_len=PRED_LEN)
    input_dim = X.shape[-1]

    n = len(X)
    train_idx, val_idx = _train_val_indices(n, train_fraction=0.8, seed=SEED)

    batch_size = 256 if torch.cuda.is_available() else 64
    train_loader = DataLoader(
        _IndexedWindowDataset(X, Y, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        _IndexedWindowDataset(X, Y, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

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
    ).to(device)

    # Channel weights: focus ONLY on target (index 0). Others are zero.
    channel_w = torch.zeros(input_dim, device=device, dtype=torch.float32)
    channel_w[0] = 1.0

    def weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d2 = (pred - target) ** 2
        # mean over batch/time/features; only target channel contributes
        return (d2 * channel_w.view(1, 1, -1)).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = weighted_mse(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            preds, targets = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_loss_sum += float(weighted_mse(out, yb).item())
                    val_batches += 1
                    preds.append(out[:, :, :1])     # only target channel
                    targets.append(yb[:, :, :1])
            val_loss = val_loss_sum / max(val_batches, 1)
            pred_cat = torch.cat(preds, dim=0)
            targ_cat = torch.cat(targets, dim=0)
            metrics = compute_metrics(pred_cat, targ_cat, continuous_cols=[0], mape_cols=[0])
            print(
                f"[{label}] epoch {epoch:3d} | train={train_loss:.6f} val={val_loss:.6f} "
                f"| MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} MAPE={metrics['MAPE']:.2f}%"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 10
                if epochs_no_improve >= PATIENCE:
                    print(f"[{label}] early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, preprocessor, features


def _train_patchtst_cuaca(
    df,
    device: torch.device,
):
    Xc, Yc, pre_cuaca, features_c = build_cuaca_dataset(df=df, input_len=INPUT_LEN, pred_len=PRED_LEN)

    n = len(Xc)
    train_idx, val_idx = _train_val_indices(n, train_fraction=0.8, seed=SEED)

    y_flat = Yc[train_idx].reshape(-1).astype(np.int64)
    counts = np.bincount(y_flat, minlength=3).astype(np.float64)
    total_labels = float(len(y_flat))
    ce_w = total_labels / (3.0 * np.maximum(counts, 1.0))
    ce_w = ce_w * (3.0 / ce_w.sum())
    weight_t = torch.tensor(ce_w, dtype=torch.float32, device=device)

    batch_size = 128 if torch.cuda.is_available() else 64
    train_loader = DataLoader(
        _IndexedWindowDataset(Xc, Yc, train_idx, y_dtype=torch.long),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        _IndexedWindowDataset(Xc, Yc, val_idx, y_dtype=torch.long),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = CuacaPatchTST(input_dim=Xc.shape[-1]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss_sum += float(criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)).item())
                    val_batches += 1
                    pred = torch.argmax(logits, dim=-1)
                    correct += int((pred == yb).sum().item())
                    total += int(yb.numel())
            val_loss = val_loss_sum / max(val_batches, 1)
            val_acc = 100.0 * correct / max(total, 1)
            print(f"[cuaca] epoch {epoch:3d} | train={train_loss:.6f} val={val_loss:.6f} | ACC={val_acc:.2f}%")

            if val_loss < best_val:
                best_val = val_loss
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 10
                if epochs_no_improve >= PATIENCE:
                    print(f"[cuaca] early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "model_state_dict": model.state_dict(),
        "input_dim": Xc.shape[-1],
        "num_classes": 3,
        "best_val_loss": best_val,
        "best_val_acc": best_acc,
        "cuaca_train_label_counts": counts.tolist(),
        "cuaca_ce_weight": ce_w.tolist(),
    }
    return meta, pre_cuaca, features_c


def main():
    parser = argparse.ArgumentParser(description="Train Multi_Model PatchTST regression + cuaca classifier.")
    parser.add_argument(
        "--cuaca-only",
        action="store_true",
        help="Skip the 5 regression models; train and save only the cuaca classifier.",
    )
    args = parser.parse_args()

    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"INPUT_LEN={INPUT_LEN}, PRED_LEN={PRED_LEN}, patch_len={PATCH_LEN}, stride={STRIDE}")

    df = load_historical_df()

    out_dir = artifacts_dir(ROOT)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Train 5 regression models ----
    if not args.cuaca_only:
        for label in REGRESSION_LABELS:
            print("\n" + "=" * 72)
            print(f"Training PatchTST regression model for label: {label}")
            print("=" * 72)
            model, preprocessor, features = _train_patchtst_single_label(label, df=df, device=device)

            art = artifacts_for_label(ROOT, label)
            torch.save(model.state_dict(), art.model_path)
            joblib.dump(preprocessor, art.preprocessor_path)
            print(f"[{label}] saved model: {art.model_path}")
            print(f"[{label}] saved preprocessor: {art.preprocessor_path}")
            print(f"[{label}] features: {features}")
    else:
        print("\n(--cuaca-only) Skipping regression models.\n")

    # ---- Train cuaca classifier (PatchTST classification) ----
    print("\n" + "=" * 72)
    print("Training PatchTST cuaca classifier (3 classes, multi-step)")
    print("=" * 72)
    cuaca_payload, pre_cuaca, features_c = _train_patchtst_cuaca(df=df, device=device)
    art_c = artifacts_for_label(ROOT, "cuaca")
    torch.save(cuaca_payload, art_c.model_path)
    joblib.dump(pre_cuaca, art_c.preprocessor_path)
    print(f"[cuaca] saved model: {art_c.model_path}")
    print(f"[cuaca] saved preprocessor: {art_c.preprocessor_path}")
    print(f"[cuaca] features: {features_c}")

    print("\nDone training 6 models." if not args.cuaca_only else "\nDone training cuaca-only.")


if __name__ == "__main__":
    main()


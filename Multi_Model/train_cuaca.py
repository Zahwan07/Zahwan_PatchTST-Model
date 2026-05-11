"""
Train only the cuaca classifier with PatchTST (3-class, multi-step).

REFERENCE-ONLY NOTE:
  The primary active training path is now:
    python Multi_Model/train_multi.py
  This file is kept for experimentation/reference and is not required in the
  main 6-model workflow.

Target classes:
  0 clear, 1 cloudy, 2 rain

Artifacts (same paths as multi_config):
  Main_model/Multi_Model/artifacts/patchtst_cuaca.pth
  Main_model/Multi_Model/artifacts/preprocessor_cuaca.joblib

Run from project root:
  python Multi_Model/train_cuaca.py
  python Multi_Model/train_cuaca.py --max-samples 12000 --epochs 80
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Multi_Model.cuaca_patchtst import CuacaPatchTST
from Multi_Model.datasets_multi import build_cuaca_dataset, load_historical_df
from Multi_Model.multi_config import INPUT_LEN, PRED_LEN, PATCH_LEN, STRIDE, artifacts_dir, artifacts_for_label


def main():
    ap = argparse.ArgumentParser(description="Train cuaca classifier only.")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=12_000,
        help="Maximum training windows (random subsample if dataset is larger). Default 12000.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"INPUT_LEN={INPUT_LEN}, PRED_LEN={PRED_LEN} (same windowing as PatchTST)")
    df = load_historical_df()
    Xc, Yc, pre_cuaca, features_c = build_cuaca_dataset(df=df, input_len=INPUT_LEN, pred_len=PRED_LEN)
    n = len(Xc)
    if args.max_samples <= 0:
        print(f"[cuaca] using all {n} windows")
    elif n > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=args.max_samples, replace=False)
        Xc, Yc = Xc[idx], Yc[idx]
        print(f"[cuaca] subsampled {args.max_samples} / {n} windows")

    X_tensor = torch.tensor(Xc, dtype=torch.float32)
    Y_tensor = torch.tensor(Yc, dtype=torch.long)
    n2 = len(X_tensor)
    idx_all = torch.randperm(n2)
    train_end = int(0.7 * n2)
    X_train, X_val = X_tensor[idx_all[:train_end]], X_tensor[idx_all[train_end:]]
    Y_train, Y_val = Y_tensor[idx_all[:train_end]], Y_tensor[idx_all[train_end:]]

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.batch_size)

    model = CuacaPatchTST(
        input_dim=Xc.shape[-1],
        input_len=INPUT_LEN,
        pred_len=PRED_LEN,
        num_classes=3,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        d_model=128,
        n_heads=16,
        num_layers=3,
        dropout=0.2,
        revin=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    best_acc = 0.0
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)  # (B, pred_len, C)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
                val_loss_sum += float(loss.item())
                val_batches += 1
                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        val_loss = val_loss_sum / max(val_batches, 1)
        val_acc = 100.0 * correct / max(total, 1)

        print(f"[cuaca] epoch {epoch:3d} | train={train_loss:.6f} val={val_loss:.6f} | acc={val_acc:.2f}%")

        if val_loss < best_val:
            best_val = val_loss
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[cuaca] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = artifacts_dir(ROOT)
    os.makedirs(out_dir, exist_ok=True)
    art_c = artifacts_for_label(ROOT, "cuaca")
    joblib.dump(pre_cuaca, art_c.preprocessor_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": Xc.shape[-1],
            "num_classes": 3,
            "best_val_loss": best_val,
            "best_val_acc": best_acc,
        },
        art_c.model_path,
    )
    print(f"[cuaca] saved PatchTST classifier: {art_c.model_path}")
    print(f"[cuaca] saved preprocessor: {art_c.preprocessor_path}")
    print(f"[cuaca] features: {features_c}")
    print("Done.")


if __name__ == "__main__":
    main()

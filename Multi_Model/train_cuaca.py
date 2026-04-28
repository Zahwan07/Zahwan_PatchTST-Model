"""
Train only the cuaca (weather state) classifier for Multi_Model.

Uses sklearn MultiOutputClassifier(LogisticRegression) — one logistic model per
forecast hour (PRED_LEN steps). L-BFGS warnings come from sklearn, not PatchTST.

For speed, training windows can be subsampled (--max-samples). Full-data fit is
very slow (720 estimators × large N).

Artifacts (same paths as multi_config):
  Main_model/Multi_Model/artifacts/cuaca_clf.joblib
  Main_model/Multi_Model/artifacts/preprocessor_cuaca.joblib

Run from project root:
  python Multi_Model/train_cuaca.py
  python Multi_Model/train_cuaca.py --max-samples 8000
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Multi_Model.cuaca_classifier import build_default_cuaca_estimator, fit_cuaca_classifier
from Multi_Model.datasets_multi import build_cuaca_dataset, load_historical_df
from Multi_Model.multi_config import INPUT_LEN, PRED_LEN, artifacts_dir, artifacts_for_label


def main():
    ap = argparse.ArgumentParser(description="Train cuaca classifier only.")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=12_000,
        help="Maximum training windows (random subsample if dataset is larger). Default 12000.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=2000, help="LogisticRegression max_iter per step.")
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for MultiOutputClassifier (-1 = all CPUs).",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"INPUT_LEN={INPUT_LEN}, PRED_LEN={PRED_LEN} (same windowing as PatchTST)")
    df = load_historical_df()
    Xc, Yc, pre_cuaca, features_c = build_cuaca_dataset(df=df, input_len=INPUT_LEN, pred_len=PRED_LEN)
    n = len(Xc)
    if args.max_samples <= 0:
        print(f"[cuaca] using all {n} windows (slow: 720 estimators × full data)")
    elif n > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=args.max_samples, replace=False)
        Xc, Yc = Xc[idx], Yc[idx]
        print(f"[cuaca] subsampled {args.max_samples} / {n} windows (use --max-samples 0 for full data; very slow)")

    clf = build_default_cuaca_estimator(max_iter=args.max_iter, n_jobs=args.n_jobs)
    print(f"[cuaca] fitting MultiOutputClassifier (PRED_LEN={PRED_LEN} outputs)...")
    clf = fit_cuaca_classifier(clf, Xc, Yc)

    out_dir = artifacts_dir(ROOT)
    os.makedirs(out_dir, exist_ok=True)
    art_c = artifacts_for_label(ROOT, "cuaca")
    joblib.dump(pre_cuaca, art_c.preprocessor_path)
    joblib.dump(clf, art_c.model_path)
    print(f"[cuaca] saved classifier: {art_c.model_path}")
    print(f"[cuaca] saved preprocessor: {art_c.preprocessor_path}")
    print(f"[cuaca] features: {features_c}")
    print("Done.")


if __name__ == "__main__":
    main()

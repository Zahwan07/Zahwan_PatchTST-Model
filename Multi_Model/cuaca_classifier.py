"""
Cuaca classifier for Multi_Model.

We treat cuaca as a 3-class classification problem (0 clear, 1 cloudy, 2 rain),
and predict multi-step horizon (PRED_LEN steps) using a MultiOutputClassifier.

Input per sample: flattened (INPUT_LEN * D) feature window.
Target per sample: vector length PRED_LEN with integer classes.
"""

from __future__ import annotations

import numpy as np


def build_default_cuaca_estimator(*, max_iter: int = 2000, n_jobs: int | None = None):
    """
    Default estimator: one LogisticRegression per horizon step (PRED_LEN outputs).
    L-BFGS is sklearn's solver for each step — not PatchTST.

    MultiOutputClassifier parallelizes across steps when n_jobs != 1.
    Higher max_iter reduces convergence warnings on noisy windows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    base = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    nj = -1 if n_jobs is None else n_jobs
    return MultiOutputClassifier(base, n_jobs=nj)


def fit_cuaca_classifier(clf, X: np.ndarray, Y: np.ndarray):
    """
    X: (N, INPUT_LEN, D) scaled
    Y: (N, PRED_LEN) int labels
    """
    X2 = X.reshape(X.shape[0], -1)
    clf.fit(X2, Y)
    return clf


def predict_cuaca(clf, X_last_window: np.ndarray) -> np.ndarray:
    """
    Predict cuaca for a single input window.
    X_last_window: (INPUT_LEN, D) scaled
    returns: (PRED_LEN,) int
    """
    X2 = X_last_window.reshape(1, -1)
    y = clf.predict(X2)
    return np.asarray(y).reshape(-1).astype(int)


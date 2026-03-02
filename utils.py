"""Evaluation metrics for PatchTST forecasts."""
import torch
import numpy as np


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.abs(pred - target).mean().item()


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error."""
    return ((pred - target) ** 2).mean().item()


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Classic MAPE (%). Explodes when target is near 0 (e.g. cuaca=0). Use smape() for a bounded metric."""
    denom = torch.abs(target) + eps
    return (torch.abs(pred - target) / denom).mean().item() * 100.0


def smape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Symmetric MAPE (%). Bounded 0–200%; safe when target or pred is near zero (e.g. cuaca)."""
    num = torch.abs(pred - target)
    denom = (torch.abs(pred) + torch.abs(target)) / 2.0 + eps
    return (num / denom).mean().item() * 100.0


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, continuous_cols: list = None) -> dict:
    """
    Return dict with MAE, MSE, MAPE (symmetric, bounded).
    If continuous_cols is given (e.g. [0,1,2] for suhu, kelembapan, ph when cuaca is one-hot),
    MAPE is computed only on those columns — one-hot columns distort MAPE.
    """
    mae_val = mae(pred, target)
    mse_val = mse(pred, target)
    if continuous_cols is not None:
        pred_cont = pred[..., continuous_cols]
        target_cont = target[..., continuous_cols]
        mape_val = smape(pred_cont, target_cont)
    else:
        mape_val = smape(pred, target)
    return {"MAE": mae_val, "MSE": mse_val, "MAPE": mape_val}

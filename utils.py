"""Evaluation metrics for PatchTST forecasts."""
import torch
import numpy as np


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.abs(pred - target).mean().item()


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error."""
    return ((pred - target) ** 2).mean().item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error (same units as target)."""
    return (mse(pred, target) ** 0.5)


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Classic MAPE (%). Explodes when target is near 0 (e.g. cuaca=0). Use smape() for a bounded metric."""
    denom = torch.abs(target) + eps
    return (torch.abs(pred - target) / denom).mean().item() * 100.0


def smape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Symmetric MAPE (%). Bounded 0–200%; safe when target or pred is near zero (e.g. cuaca)."""
    num = torch.abs(pred - target)
    denom = (torch.abs(pred) + torch.abs(target)) / 2.0 + eps
    return (num / denom).mean().item() * 100.0


def _contiguous_slice(cols: list):
    """If cols is a contiguous range (e.g. [0,1,2,3,4]), return slice(0, 5). Else None. Avoids tensor copy."""
    if not cols:
        return None
    if cols == list(range(cols[0], cols[-1] + 1)):
        return slice(cols[0], cols[-1] + 1)
    return None


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    continuous_cols: list = None,
    mape_cols: list = None,
) -> dict:
    """
    Return dict with MAE, MSE, RMSE, MAPE (SMAPE %).
    - If continuous_cols is given: MAE, MSE, RMSE are computed only on those channels
      (so one-hot / categorical channels don't inflate RMSE).
    - mape_cols: if given, SMAPE is computed only on these (e.g. [0,1,3] = suhu, humidity, ph
      so MAPE is interpretable and not dominated by zero-heavy light/precipitation). Else uses continuous_cols.
    Uses slices for contiguous column ranges to avoid slow tensor copies during validation.
    """
    if continuous_cols is not None:
        sl = _contiguous_slice(continuous_cols)
        if sl is not None:
            pred_cont = pred[..., sl]
            target_cont = target[..., sl]
        else:
            pred_cont = pred[..., continuous_cols]
            target_cont = target[..., continuous_cols]
        mae_val = mae(pred_cont, target_cont)
        mse_val = mse(pred_cont, target_cont)
        rmse_val = rmse(pred_cont, target_cont)
        if mape_cols is not None:
            sl_mape = _contiguous_slice(mape_cols)
            if sl_mape is not None:
                pred_mape = pred[..., sl_mape]
                target_mape = target[..., sl_mape]
            else:
                pred_mape = pred[..., mape_cols]
                target_mape = target[..., mape_cols]
            mape_val = smape(pred_mape, target_mape)
        else:
            mape_val = smape(pred_cont, target_cont)
    else:
        mae_val = mae(pred, target)
        mse_val = mse(pred, target)
        rmse_val = rmse(pred, target)
        mape_val = smape(pred, target)
    return {"MAE": mae_val, "MSE": mse_val, "RMSE": rmse_val, "MAPE": mape_val}

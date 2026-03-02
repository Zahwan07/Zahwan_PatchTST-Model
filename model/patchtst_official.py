"""
Official PatchTST from "A Time Series is Worth 64 Words" (Nie et al.).

Wraps the PatchTST-main implementation for use in the Main_model pipeline.
Expects input (batch, seq_len, channels) and outputs (batch, pred_len, channels).

Requires: PatchTST-main folder in project root (or parent directory).
"""
import os
import sys

# Add PatchTST_supervised to path so we can import their Model
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHTST_SUP = os.path.join(ROOT, "PatchTST-main", "PatchTST_supervised")
if not os.path.isdir(PATCHTST_SUP):
    PATCHTST_SUP = os.path.join(os.path.dirname(ROOT), "PatchTST-main", "PatchTST_supervised")
if PATCHTST_SUP not in sys.path:
    sys.path.insert(0, PATCHTST_SUP)

import torch
import torch.nn as nn
from types import SimpleNamespace


def build_patchtst_config(
    enc_in: int = 4,
    seq_len: int = 168,
    pred_len: int = 720,
    patch_len: int = 16,
    stride: int = 8,
    padding_patch: str = "end",
    revin: bool = True,
    affine: bool = False,
    subtract_last: bool = False,
    decomposition: bool = False,
    kernel_size: int = 25,
    individual: bool = False,
    d_model: int = 128,
    n_heads: int = 16,
    e_layers: int = 3,
    d_ff: int = 256,
    dropout: float = 0.2,
    fc_dropout: float = 0.2,
    head_dropout: float = 0.0,
) -> SimpleNamespace:
    """Build config namespace for official PatchTST Model."""
    return SimpleNamespace(
        enc_in=enc_in,
        seq_len=seq_len,
        pred_len=pred_len,
        patch_len=patch_len,
        stride=stride,
        padding_patch=padding_patch,
        revin=revin,
        affine=affine,
        subtract_last=subtract_last,
        decomposition=decomposition,
        kernel_size=kernel_size,
        individual=individual,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=dropout,
        fc_dropout=fc_dropout,
        head_dropout=head_dropout,
    )


class PatchTST_Official(nn.Module):
    """
    Official PatchTST with patching for long-term forecasting.
    Input: (batch, seq_len, channels), Output: (batch, pred_len, channels)
    """

    def __init__(
        self,
        input_dim: int,
        input_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        num_layers: int = 3,
        dropout: float = 0.2,
        revin: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.pred_len = pred_len

        configs = build_patchtst_config(
            enc_in=input_dim,
            seq_len=input_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=num_layers,
            dropout=dropout,
            revin=revin,
            **kwargs,
        )
        from models.PatchTST import Model
        self._model = Model(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) — already correct format for official model
        return self._model(x)

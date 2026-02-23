"""
Wrapper around the official PatchTST from GitHub (PatchTST-main/PatchTST_supervised).
Uses patching, RevIN, and channel-independent backbone for better time-series
forecasting. Same interface as model.patchtst for drop-in use in train.py/predict.py.
"""
import os
import sys

# Add official PatchTST repo to path so "from layers.xxx" works
_PATCHTST_DIR = os.path.join(os.path.dirname(__file__), "..", "PatchTST-main", "PatchTST_supervised")
_PATCHTST_DIR = os.path.abspath(_PATCHTST_DIR)
if os.path.isdir(_PATCHTST_DIR) and _PATCHTST_DIR not in sys.path:
    sys.path.insert(0, _PATCHTST_DIR)

import torch
import torch.nn as nn


def _make_config(enc_in, seq_len, pred_len, patch_len=4, stride=2, d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.15, revin=True):
    """Build a config object for the official PatchTST Model."""
    class C:
        pass
    c = C()
    c.enc_in = enc_in
    c.seq_len = seq_len
    c.pred_len = pred_len
    c.patch_len = patch_len
    c.stride = stride
    c.padding_patch = "end"
    c.d_model = d_model
    c.n_heads = n_heads
    c.e_layers = e_layers
    c.d_ff = d_ff
    c.dropout = dropout
    c.fc_dropout = 0.05
    c.head_dropout = 0.1
    c.revin = bool(revin)
    c.affine = False
    c.subtract_last = False
    c.decomposition = False
    c.kernel_size = 25
    c.individual = False
    c.norm = "BatchNorm"
    c.attn_dropout = 0.0
    c.act = "gelu"
    c.key_padding_mask = "auto"
    c.padding_var = None
    c.attn_mask = None
    c.res_attention = True
    c.pre_norm = False
    c.store_attn = False
    c.pe = "zeros"
    c.learn_pe = True
    c.pretrain_head = False
    c.head_type = "flatten"
    c.verbose = False
    return c


class PatchTST(nn.Module):
    """
    Official PatchTST: patching, RevIN, channel-independent encoder.
    Input: (batch, seq_len, nvars) e.g. (B, 24, 4)
    Output: (batch, pred_len, nvars) e.g. (B, 7, 4)
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_layers=2,
        pred_len=7,
        dropout=0.15,
        seq_len=24,
        patch_len=4,
        stride=2,
        revin=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len
        self._seq_len = seq_len

        if not os.path.isdir(_PATCHTST_DIR):
            raise FileNotFoundError(
                f"Official PatchTST not found at {_PATCHTST_DIR}. "
                "Use model.patchtst (simple Transformer) instead, or add PatchTST-main folder."
            )

        from models.PatchTST import Model as _PatchTSTModel

        configs = _make_config(
            enc_in=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=num_layers,
            d_ff=max(d_model * 2, 128),
            dropout=dropout,
            revin=revin,
        )
        self._model = _PatchTSTModel(configs)

    def forward(self, x):
        # x: (B, seq_len, nvars) -> official model expects same, returns (B, pred_len, nvars)
        return self._model(x)

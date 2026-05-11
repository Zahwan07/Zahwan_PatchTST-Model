"""
REFERENCE-ONLY helper for cuaca PatchTST variants.

Primary active workflow trains all 6 labels via:
  python Multi_Model/train_multi.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.patchtst_official import PatchTST_Official


class CuacaPatchTST(nn.Module):
    """
    Multi-step cuaca classifier on top of PatchTST backbone.
    Input:  (B, INPUT_LEN, D)
    Output: logits (B, PRED_LEN, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        input_len: int,
        pred_len: int,
        num_classes: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        num_layers: int = 3,
        dropout: float = 0.2,
        revin: bool = True,
    ):
        super().__init__()
        self.backbone = PatchTST_Official(
            input_dim=input_dim,
            input_len=input_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            revin=revin,
        )
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)               # (B, pred_len, input_dim)
        logits = self.classifier(feat)        # (B, pred_len, C)
        return logits


def predict_cuaca_classes(model: nn.Module, x_last_window: torch.Tensor) -> torch.Tensor:
    """
    x_last_window: (1, INPUT_LEN, D) float tensor
    returns: (PRED_LEN,) class indices
    """
    with torch.no_grad():
        logits = model(x_last_window)
        cls = torch.argmax(logits, dim=-1)   # (1, pred_len)
    return cls.squeeze(0)

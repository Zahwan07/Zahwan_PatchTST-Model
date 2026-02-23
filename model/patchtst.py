import torch
import torch.nn as nn

class PatchTST(nn.Module):
    """Predicts next pred_len timesteps for all input_dim channels (e.g. 7 days × 4 vars)."""
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, pred_len=7, dropout=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len

        self.embedding = nn.Linear(input_dim, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, pred_len * input_dim)
        self.head_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed_dropout(self.embedding(x))
        x = self.transformer(x)
        x = x[:, -1, :]
        out = self.head(self.head_dropout(x))
        return out.view(-1, self.pred_len, self.input_dim)

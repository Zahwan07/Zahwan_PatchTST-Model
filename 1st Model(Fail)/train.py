import torch
import torch.nn as nn
import joblib
import random
import numpy as np
from preprocess import prepare_data, create_dataset
from model.patchtst import PatchTST
from config import INPUT_DIM, INPUT_LEN, PRED_LEN
from utils import compute_metrics

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Loading and preparing data...")
data_scaled, scaler = prepare_data()
X, Y = create_dataset(data_scaled, input_len=INPUT_LEN, pred_len=PRED_LEN)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)  # (batch, pred_len, input_dim) full env forecast

# Train / validation split (80% / 20%)
n = len(X_tensor)
idx = torch.randperm(n)
train_end = int(0.8 * n)
X_train, X_val = X_tensor[idx[:train_end]], X_tensor[idx[train_end:]]
Y_train, Y_val = Y_tensor[idx[:train_end]], Y_tensor[idx[train_end:]]
print(f"Samples: train={len(X_train)}, val={len(X_val)}")

model = PatchTST(input_dim=INPUT_DIM, pred_len=PRED_LEN, dropout=0.15)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

EPOCHS = 500
PATIENCE = 50
best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, Y_val)
            metrics = compute_metrics(val_out, Y_val)
        print(f"Epoch {epoch:3d}, train_loss={loss.item():.6f}, val_loss={val_loss.item():.6f} | val MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.6f}, MAPE={metrics['MAPE']:.2f}%")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 10 if epoch % 10 == 0 else 1

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    print("Restored best model (lowest validation loss).")

model.eval()
with torch.no_grad():
    val_out = model(X_val)
    final_val = criterion(val_out, Y_val)
    final_metrics = compute_metrics(val_out, Y_val)

print("\n--- Validation metrics (best model) ---")
print(f"Loss (MSE): {final_val.item():.6f}")
print(f"MAE:  {final_metrics['MAE']:.4f}")
print(f"MSE: {final_metrics['MSE']:.6f}")
print(f"MAPE: {final_metrics['MAPE']:.2f}%")

torch.save(model.state_dict(), "patchtst_model.pth")
joblib.dump(scaler, "scaler.joblib")
print("Model (patchtst_model.pth) and scaler (scaler.joblib) saved.")

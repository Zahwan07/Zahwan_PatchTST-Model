import torch
import pandas as pd
import joblib
from model.patchtst import PatchTST
from config import FEATURES, INPUT_DIM, PRED_LEN, INPUT_LEN

# Load model and same scaler used in training
model = PatchTST(input_dim=INPUT_DIM, pred_len=PRED_LEN)
model.load_state_dict(torch.load("patchtst_model.pth"))
model.eval()
scaler = joblib.load("scaler.joblib")

# Load realtime data
df = pd.read_csv("data/realtime_data.csv")
data = df[FEATURES].values
data_scaled = scaler.transform(data)  # use transform only (no fit)

# Use last INPUT_LEN timesteps as input window
input_window = data_scaled[-INPUT_LEN:]

input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    prediction = model(input_tensor)

print(f"Prediction (next {PRED_LEN} timesteps):")
print(prediction.numpy())

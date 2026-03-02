import numpy as np
from sklearn.preprocessing import StandardScaler
from fusion import load_and_merge
from config import FEATURES

def prepare_data():
    df = load_and_merge()
    data = df[FEATURES].values

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler


def create_dataset(data, input_len=24, pred_len=6):
    X, Y = [], []

    for i in range(len(data) - input_len - pred_len):
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+pred_len])

    return np.array(X), np.array(Y)

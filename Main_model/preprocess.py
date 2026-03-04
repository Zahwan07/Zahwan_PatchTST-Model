"""
Preprocess for Main_model pipeline: load data/historical_environment.csv only.
Uses hourly data, 1-week input, 1-month forecast (config_exp).

Cuaca is one-hot encoded (categorical); suhu, kelembapan, ph are continuous (StandardScaler).
"""
import os
import numpy as np
import pandas as pd

from Main_model.config_exp import INPUT_LEN, PRED_LEN
from Main_model.preprocessor import EnvironmentPreprocessor

# Project root (parent of Main_model)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_data():
    path = os.path.join(ROOT, "data", "historical_environment.csv")
    df = pd.read_csv(path)
    preprocessor = EnvironmentPreprocessor()
    preprocessor.fit(df)
    data_scaled = preprocessor.transform(df)

    return data_scaled, preprocessor


def create_dataset(data, input_len=None, pred_len=None):
    input_len = input_len or INPUT_LEN
    pred_len = pred_len or PRED_LEN
    X, Y = [], []

    for i in range(len(data) - input_len - pred_len):
        X.append(data[i : i + input_len])
        Y.append(data[i + input_len : i + input_len + pred_len])

    return np.array(X), np.array(Y)

"""
Preprocessor for Main_model pipeline: cuaca as one-hot categorical,
suhu/kelembapan/ph as continuous (StandardScaler).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class EnvironmentPreprocessor:
    """
    Transforms: [suhu, cuaca, kelembapan, ph] -> [suhu_scaled, kelembapan_scaled, ph_scaled, cuaca_0, cuaca_1, cuaca_2]
    Inverse: 6 columns -> [suhu, cuaca, kelembapan, ph]
    """

    CONTINUOUS_COLS = ["suhu", "kelembapan", "ph"]
    CATEGORICAL_COL = "cuaca"

    def __init__(self):
        self.scaler_ = StandardScaler()
        self.cuaca_classes_ = None  # sorted unique cuaca values, e.g. [0, 1, 2]

    def fit(self, df: pd.DataFrame):
        """Fit on raw data with columns suhu, cuaca, kelembapan, ph."""
        self.cuaca_classes_ = sorted(df[self.CATEGORICAL_COL].unique().astype(int))
        cont = df[self.CONTINUOUS_COLS].values
        self.scaler_.fit(cont)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform to [suhu_scaled, kelembapan_scaled, ph_scaled, cuaca_0, cuaca_1, cuaca_2]."""
        cont = df[self.CONTINUOUS_COLS].values
        cat = df[self.CATEGORICAL_COL].values.astype(int)
        cont_scaled = self.scaler_.transform(cont)
        cat_onehot = np.eye(len(self.cuaca_classes_))[np.searchsorted(self.cuaca_classes_, cat)]
        return np.concatenate([cont_scaled, cat_onehot], axis=1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform model output (6 cols) -> [suhu, cuaca, kelembapan, ph].
        X: (..., 6) with [suhu_scaled, kelembapan_scaled, ph_scaled, cuaca_0, cuaca_1, cuaca_2]
        """
        n_cat = len(self.cuaca_classes_)
        n_cont = 3
        shape = X.shape
        X_flat = X.reshape(-1, n_cont + n_cat)
        cont_scaled = X_flat[:, :n_cont]
        cat_onehot = X_flat[:, n_cont:]
        cont_raw = self.scaler_.inverse_transform(cont_scaled)
        cat_idx = np.argmax(cat_onehot, axis=1)
        cuaca_raw = np.array([self.cuaca_classes_[i] for i in cat_idx])
        out = np.zeros((X_flat.shape[0], 4))
        out[:, 0] = cont_raw[:, 0]   # suhu
        out[:, 1] = cuaca_raw
        out[:, 2] = cont_raw[:, 1]   # kelembapan
        out[:, 3] = cont_raw[:, 2]   # ph
        return out.reshape(*shape[:-1], 4)

    @property
    def n_features_in_(self) -> int:
        """Number of encoded features (for model input_dim)."""
        return 3 + len(self.cuaca_classes_)

"""
Preprocessor for Main_model pipeline: cuaca as one-hot categorical,
all other numeric columns as continuous (StandardScaler).

Supports expanded features: suhu, kelembapan, ph, dewpoint, pressure, cloud_cover,
shortwave_radiation, wind, hour/doy cyclical, temp_lags, etc.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Main_model.config_exp import ALL_FEATURES


class EnvironmentPreprocessor:
    """
    Transforms: [continuous_cols..., cuaca] -> [continuous_scaled..., cuaca_0, cuaca_1, cuaca_2]
    Inverse: encoded -> [suhu, cuaca, kelembapan, ph, ...] (all columns)
    """

    CATEGORICAL_COL = "cuaca"

    def __init__(self):
        self.scaler_ = StandardScaler()
        self.cuaca_classes_ = None
        self.continuous_cols_ = None  # list of col names (all except cuaca)

    def fit(self, df: pd.DataFrame):
        """Fit on raw data. Uses ALL_FEATURES present in df."""
        present = [c for c in ALL_FEATURES if c in df.columns]
        self.continuous_cols_ = [c for c in present if c != self.CATEGORICAL_COL]
        self.cuaca_classes_ = sorted(df[self.CATEGORICAL_COL].unique().astype(int))
        cont = df[self.continuous_cols_].values
        self.scaler_.fit(cont)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform to [continuous_scaled..., cuaca_onehot]."""
        cont = df[self.continuous_cols_].values
        cat = df[self.CATEGORICAL_COL].values.astype(int)
        cont_scaled = self.scaler_.transform(cont)
        cat_onehot = np.eye(len(self.cuaca_classes_))[np.searchsorted(self.cuaca_classes_, cat)]
        return np.concatenate([cont_scaled, cat_onehot], axis=1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform model output -> [col1, col2, ..., cuaca, ...].
        Returns full feature set in order: continuous_cols (with cuaca inserted at original position).
        """
        n_cat = len(self.cuaca_classes_)
        n_cont = len(self.continuous_cols_)
        shape = X.shape
        X_flat = X.reshape(-1, n_cont + n_cat)
        cont_scaled = X_flat[:, :n_cont]
        cat_onehot = X_flat[:, n_cont:]
        cont_raw = self.scaler_.inverse_transform(cont_scaled)
        cat_idx = np.argmax(cat_onehot, axis=1)
        cuaca_raw = np.array([self.cuaca_classes_[i] for i in cat_idx])

        n_out = len(ALL_FEATURES)
        out = np.zeros((X_flat.shape[0], n_out))
        cont_idx = 0
        cat_placed = False
        for i, col in enumerate(ALL_FEATURES):
            if col == self.CATEGORICAL_COL:
                out[:, i] = cuaca_raw
                cat_placed = True
            elif col in self.continuous_cols_:
                out[:, i] = cont_raw[:, self.continuous_cols_.index(col)]
            # else: col not in training data, leave 0
        return out.reshape(*shape[:-1], n_out)

    @property
    def n_features_in_(self) -> int:
        """Number of encoded features (for model input_dim)."""
        return len(self.continuous_cols_) + len(self.cuaca_classes_)
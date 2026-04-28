"""
Preprocessor for Main_model pipeline: all numeric columns as continuous (StandardScaler),
or legacy mode with cuaca one-hot if "cuaca" is listed in ALL_FEATURES.

Supports expanded features: suhu, humidity, light_intensity, ph, dewpoint, pressure,
cloud_cover, wind, hour/doy cyclical, temp_lags (hydroponic; no soil moisture).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Main_model.config_exp import ALL_FEATURES


class EnvironmentPreprocessor:
    """
    Transforms: either [continuous_scaled] only, or legacy [continuous_scaled..., cuaca_onehot].
    Inverse:
      - default: encoded -> full columns in ALL_FEATURES order (backward compatible)
      - subset mode: encoded -> only selected features (in that exact order)
    """

    CATEGORICAL_COL = "cuaca"

    def __init__(self, features: list[str] | None = None, inverse_output: str = "all"):
        """
        features:
          - None: use ALL_FEATURES (default pipeline)
          - list[str]: use only these features (for Multi_Model per-label training)

        inverse_output:
          - "all": inverse_transform returns array with len(ALL_FEATURES) columns (default)
          - "subset": inverse_transform returns only selected columns (len(features)) in the given order
        """
        self.scaler_ = StandardScaler()
        self.cuaca_classes_ = None
        self.continuous_cols_ = None
        self.use_categorical_ = False
        self.features_ = features[:] if features is not None else None
        if inverse_output not in ("all", "subset"):
            raise ValueError("inverse_output must be 'all' or 'subset'")
        self.inverse_output_ = inverse_output

    def fit(self, df: pd.DataFrame):
        """Fit on raw data. Uses configured feature list present in df."""
        base = self.features_ if self.features_ is not None else ALL_FEATURES
        present = [c for c in base if c in df.columns]
        if self.CATEGORICAL_COL in ALL_FEATURES and self.CATEGORICAL_COL in present:
            self.use_categorical_ = True
            self.continuous_cols_ = [c for c in present if c != self.CATEGORICAL_COL]
            self.cuaca_classes_ = sorted(df[self.CATEGORICAL_COL].unique().astype(int))
        else:
            self.use_categorical_ = False
            self.continuous_cols_ = list(present)
            self.cuaca_classes_ = []
        cont = df[self.continuous_cols_].values
        self.scaler_.fit(cont)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform to scaled continuous [+ optional cuaca onehot]."""
        cont = df[self.continuous_cols_].values
        cont_scaled = self.scaler_.transform(cont)
        if self.use_categorical_:
            cat = df[self.CATEGORICAL_COL].values.astype(int)
            cat_onehot = np.eye(len(self.cuaca_classes_))[np.searchsorted(self.cuaca_classes_, cat)]
            return np.concatenate([cont_scaled, cat_onehot], axis=1)
        return cont_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform model output -> configured output columns."""
        n_cont = len(self.continuous_cols_)
        shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])

        if self.use_categorical_:
            cont_scaled = X_flat[:, :n_cont]
            cat_onehot = X_flat[:, n_cont:]
            cont_raw = self.scaler_.inverse_transform(cont_scaled)
            cat_idx = np.argmax(cat_onehot, axis=1)
            cuaca_raw = np.array([self.cuaca_classes_[i] for i in cat_idx])
        else:
            cont_scaled = X_flat[:, :n_cont]
            cont_raw = self.scaler_.inverse_transform(cont_scaled)

        if self.inverse_output_ == "subset":
            # Return only selected features (or continuous_cols_ fallback) in their configured order
            out_cols = self.features_ if self.features_ is not None else list(self.continuous_cols_)
            out = np.zeros((X_flat.shape[0], len(out_cols)))
            for i, col in enumerate(out_cols):
                if self.use_categorical_ and col == self.CATEGORICAL_COL:
                    out[:, i] = cuaca_raw
                elif col in self.continuous_cols_:
                    out[:, i] = cont_raw[:, self.continuous_cols_.index(col)]
            return out.reshape(*shape[:-1], len(out_cols))

        # Backward compatible: output in ALL_FEATURES order
        n_out = len(ALL_FEATURES)
        out_all = np.zeros((X_flat.shape[0], n_out))
        for i, col in enumerate(ALL_FEATURES):
            if self.use_categorical_ and col == self.CATEGORICAL_COL:
                out_all[:, i] = cuaca_raw
            elif col in self.continuous_cols_:
                out_all[:, i] = cont_raw[:, self.continuous_cols_.index(col)]
        return out_all.reshape(*shape[:-1], n_out)

    @property
    def n_features_in_(self) -> int:
        """Number of encoded features (for model input_dim)."""
        if self.use_categorical_:
            return len(self.continuous_cols_) + len(self.cuaca_classes_)
        return len(self.continuous_cols_)

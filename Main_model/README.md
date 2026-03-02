# Main_model pipeline (hourly, 5 years, 1-month forecast)

Uses Open-Meteo **hourly** historical data (last 5 years) for **long-term** forecasting: 1 month ahead.

## Data

- **Source:** `data/historical_environment.csv` from `fetch_openmeteo_historical.py`
- **Resolution:** Hourly (≈43,800 rows for 5 years)
- **Columns:** suhu, cuaca, kelembapan, ph

## Config (Main_model/config_exp.py)

- **INPUT_LEN = 168** (1 week of hours)
- **PRED_LEN = 720** (30 days = 1 month of hours)

## Run

1. **Fetch data** (hourly, 5 years):

   ```bash
   python fetch_openmeteo_historical.py
   ```

2. **Train** (from project root):

   ```bash
   python Main_model/train2.py
   ```

## Outputs

- `Main_model/patchtst_model_exp.pth` — model for 1-month forecast
- `Main_model/preprocessor_exp.joblib` — preprocessor (cuaca one-hot + continuous scaling)

Use these files for prediction (e.g. a separate `predict_exp.py` that loads the last 168 hours and outputs 720-hour forecast).

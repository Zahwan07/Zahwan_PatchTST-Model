"""
Step 2: Compare actual realtime data with optimal ranges and send to Qwen.

- Reads the latest row from data/realtime_data.csv (today's conditions).
- Compares with optimal temp/humidity/light/pH for hydroponic red leaf lettuce (config.py).
- Sends summary to Qwen (Ollama) for recommendations. No soil moisture.

Run from project root:  python Main_model/combine_forecast_realtime.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from config import (
    LETTUCE_TEMP_OPTIMAL,
    LETTUCE_HUMIDITY_OPTIMAL,
    LETTUCE_LIGHT_OPTIMAL,
    LETTUCE_PH_OPTIMAL,
)
from llm_recommendations import get_recommendations_from_current_conditions

REALTIME_PATH = os.path.join(ROOT, "data", "realtime_data.csv")

df = pd.read_csv(REALTIME_PATH)
row = df.iloc[-1]
suhu = float(row["suhu"])
humidity = float(row.get("humidity", row.get("kelembapan", 0)))
light_intensity = float(row.get("light_intensity", 0))
ph = float(row["ph"])
cuaca = int(row["cuaca"])
cuaca_str = {0: "clear", 1: "cloudy", 2: "rain"}.get(cuaca, str(cuaca))

t_lo, t_hi = LETTUCE_TEMP_OPTIMAL
h_lo, h_hi = LETTUCE_HUMIDITY_OPTIMAL
l_lo, l_hi = LETTUCE_LIGHT_OPTIMAL
p_lo, p_hi = LETTUCE_PH_OPTIMAL

today_summary = (
    f"Today's temperature is {suhu:.1f}°C. Weather: {cuaca_str}. "
    f"Humidity {humidity:.2f} (0–1). Light intensity {light_intensity:.0f} W/m². pH is {ph:.2f}. "
    f"Optimal for red leaf lettuce (hydroponic): temperature {t_lo}–{t_hi}°C, "
    f"humidity {h_lo:.2f}–{h_hi:.2f}, light {l_lo}–{l_hi} W/m², pH {p_lo}–{p_hi}."
)

print(today_summary)
print()
recommendation = get_recommendations_from_current_conditions(today_summary)
print(recommendation)

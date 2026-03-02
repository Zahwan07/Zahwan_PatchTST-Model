"""
Generate reference dataset for red leaf lettuce (Lactuca sativa L.) with
optimal soil moisture and pH ranges. Use this when you don't yet have
sensor data for kelembapan and ph — combine with BMKG weather/temperature
(e.g. fetch_bmkg_coblong.py) or existing CSVs.

See data/lettuce_reference.md for optimal ranges and sources.
"""
import csv
import random
from pathlib import Path

from config import FEATURES, LETTUCE_KELEMBAPAN_OPTIMAL, LETTUCE_PH_OPTIMAL

# Lettuce-friendly temperature range (°C) for Bandung / Coblong
TEMP_MIN, TEMP_MAX = 20.0, 28.0
# Slight variation around optimal for realistic time series
KELEMBAPAN_LO, KELEMBAPAN_HI = LETTUCE_KELEMBAPAN_OPTIMAL
PH_LO, PH_HI = LETTUCE_PH_OPTIMAL
# cuaca: 0=cerah, 1=berawan, 2=hujan ringan
NUM_ROWS = 200


def generate_row(prev: dict | None = None) -> dict:
    """One row with lettuce-optimal kelembapan and ph, plausible suhu and cuaca."""
    if prev is None:
        suhu = random.uniform(TEMP_MIN, TEMP_MAX)
        cuaca = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
        kelembapan = random.uniform(KELEMBAPAN_LO, KELEMBAPAN_HI)
        ph = random.uniform(PH_LO, PH_HI)
    else:
        # Slight autocorrelation for time-series feel
        suhu = max(TEMP_MIN, min(TEMP_MAX, prev["suhu"] + random.gauss(0, 1.0)))
        cuaca = prev["cuaca"] if random.random() < 0.7 else random.choice([0, 1, 2])
        kelembapan = max(0.2, min(0.85, prev["kelembapan"] + random.gauss(0, 0.03)))
        kelembapan = max(KELEMBAPAN_LO, min(KELEMBAPAN_HI, kelembapan))
        ph = max(5.2, min(7.5, prev["ph"] + random.gauss(0, 0.15)))
        ph = max(PH_LO, min(PH_HI, ph))
    return {"suhu": round(suhu, 2), "cuaca": cuaca, "kelembapan": round(kelembapan, 3), "ph": round(ph, 2)}


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lettuce_reference_data.csv"

    rows = []
    prev = None
    for _ in range(NUM_ROWS):
        prev = generate_row(prev)
        rows.append(prev)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURES)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print("Kelembapan in [%.2f, %.2f], pH in [%.1f, %.1f] (Lactuca sativa L. optimal)." % (
        KELEMBAPAN_LO, KELEMBAPAN_HI, PH_LO, PH_HI))
    print("Use with fusion: copy or merge into internet_data.csv / realtime_data.csv, or load this file in preprocess.")


if __name__ == "__main__":
    main()

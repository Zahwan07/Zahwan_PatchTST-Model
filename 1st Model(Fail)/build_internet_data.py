"""
Build internet_data.csv by combining:
  - fetch_bmkg_coblong.py  → weather & temperature (Coblong, Bandung)
  - generate_lettuce_data.py → soil moisture & pH (optimal for Lactuca sativa L.)

Then writes dummy realtime_data.csv for prediction when you don't have live sensors.

Run this before training. Training uses fusion.py to merge internet_data + realtime_data.
Prediction uses realtime_data (dummy) as the input window for PatchTST.
"""
import csv
from pathlib import Path

import pandas as pd

from config import FEATURES, INPUT_LEN

DATA_DIR = Path("data")
INTERNET_CSV = DATA_DIR / "internet_data.csv"
REALTIME_CSV = DATA_DIR / "realtime_data.csv"
BMKG_CSV = DATA_DIR / "bmkg_coblong.csv"
LETTUCE_CSV = DATA_DIR / "lettuce_reference_data.csv"


def load_bmkg_rows():
    """Load BMKG data from CSV if present; else try fetch and save."""
    if BMKG_CSV.exists():
        try:
            df = pd.read_csv(BMKG_CSV)
            if len(df) > 0 and all(c in df.columns for c in FEATURES):
                return df[FEATURES].to_dict("records")
        except Exception as e:
            print(f"Warning: could not read {BMKG_CSV}: {e}")
    try:
        from fetch_bmkg_coblong import fetch_bmkg_rows
        rows = fetch_bmkg_rows(save_to_bmkg_csv=True)
        return [{"suhu": r["suhu"], "cuaca": r["cuaca"], "kelembapan": r["kelembapan"], "ph": r["ph"]} for r in rows]
    except Exception as e:
        print(f"Warning: BMKG fetch failed ({e}). Using lettuce data for suhu/cuaca too.")
    return []


def load_lettuce_rows():
    """Load or generate lettuce reference data (optimal kelembapan & pH)."""
    if LETTUCE_CSV.exists():
        try:
            df = pd.read_csv(LETTUCE_CSV)
            if len(df) > 0 and all(c in df.columns for c in FEATURES):
                return df[FEATURES].to_dict("records")
        except Exception as e:
            print(f"Warning: could not read {LETTUCE_CSV}: {e}")
    print("Generating lettuce reference data...")
    from generate_lettuce_data import main as gen_lettuce
    gen_lettuce()
    if LETTUCE_CSV.exists():
        df = pd.read_csv(LETTUCE_CSV)
        return df[FEATURES].to_dict("records")
    return []


def build_internet_data():
    """Merge BMKG (suhu, cuaca) + lettuce (kelembapan, ph) → internet_data.csv."""
    DATA_DIR.mkdir(exist_ok=True)

    bmkg = load_bmkg_rows()
    lettuce = load_lettuce_rows()
    if not lettuce:
        raise RuntimeError("No lettuce data. Run generate_lettuce_data.py first.")

    # Target length = lettuce (we have plenty). For each row: suhu, cuaca from BMKG (cycle); kelembapan, ph from lettuce.
    n = len(lettuce)
    if bmkg:
        rows = []
        for i in range(n):
            b = bmkg[i % len(bmkg)]
            l = lettuce[i]
            rows.append({
                "suhu": b["suhu"],
                "cuaca": b["cuaca"],
                "kelembapan": l["kelembapan"],
                "ph": l["ph"],
            })
        print(f"Merged BMKG ({len(bmkg)} rows) + lettuce ({n} rows) → internet_data ({n} rows).")
    else:
        rows = lettuce
        print(f"No BMKG data; using lettuce only ({n} rows) for internet_data.")

    with open(INTERNET_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURES)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {INTERNET_CSV}")


def write_dummy_realtime():
    """Write dummy realtime_data.csv for prediction (no live sensors)."""
    DATA_DIR.mkdir(exist_ok=True)
    # Need at least INPUT_LEN rows for prediction window; add extra for safety
    n_dummy = max(INPUT_LEN + 10, 40)
    # Plausible Bandung/Coblong + lettuce-optimal ranges
    import random
    random.seed(42)
    rows = []
    prev = {"suhu": 25.0, "cuaca": 1, "kelembapan": 0.55, "ph": 6.5}
    for _ in range(n_dummy):
        prev = {
            "suhu": round(max(20, min(28, prev["suhu"] + random.gauss(0, 0.8))), 2),
            "cuaca": prev["cuaca"] if random.random() < 0.7 else random.choice([0, 1, 2]),
            "kelembapan": round(max(0.35, min(0.65, prev["kelembapan"] + random.gauss(0, 0.02))), 3),
            "ph": round(max(6.0, min(7.0, prev["ph"] + random.gauss(0, 0.1))), 2),
        }
        rows.append(prev)

    with open(REALTIME_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURES)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote dummy {REALTIME_CSV} ({n_dummy} rows) for prediction input.")


def main():
    build_internet_data()
    write_dummy_realtime()
    print("Done. Next: run train.py, then predict.py (uses realtime_data as input window).")


if __name__ == "__main__":
    main()

"""
Fetch weather and temperature from BMKG for Kecamatan Coblong, Bandung (32.73.02).
Updates data/internet_data.csv or data/realtime_data.csv with new rows (suhu, cuaca, kelembapan).
See data/cuaca-suhu for API and location code reference.
"""
import os
import re
import csv
from datetime import datetime
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

# Location: Kecamatan Coblong, Kota Bandung (data/cuaca-suhu)
BMKG_COBLONG_CODE = "32.73.02"
API_URL = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={BMKG_COBLONG_CODE}"

# Map BMKG weather text to numeric code for model input
CUACA_MAP = {
    "cerah": 0,
    "berawan": 1,
    "hujan ringan": 2,
    "hujan": 2,
    "hujan sedang": 3,
    "hujan lebat": 4,
    "petir": 5,
}


def _cuaca_to_code(text):
    if not text:
        return 1  # default berawan
    t = text.lower().strip()
    for key, code in CUACA_MAP.items():
        if key in t:
            return code
    return 1


def fetch_bmkg_xml(url):
    """Fetch URL; return raw bytes or None."""
    try:
        with urlopen(url, timeout=15) as r:
            return r.read()
    except (HTTPError, URLError, OSError) as e:
        print(f"Fetch error: {e}")
        return None


def parse_xml_cuaca_suhu(raw):
    """
    Parse BMKG XML if present. Return list of dicts with keys: suhu, cuaca, kelembapan.
    If response is not XML or has no known structure, return [].
    """
    try:
        raw_str = raw.decode("utf-8", errors="ignore")
    except Exception:
        return []
    # If it's HTML (e.g. 404 page), skip
    if "<!DOCTYPE html" in raw_str or "<html" in raw_str.lower():
        return []
    rows = []
    # Simple extraction: look for temperature and weather-like tags (structure may vary by BMKG)
    # Pattern: <suhu> or t_min/t_max, and cuaca/kode_cuaca
    suhu_vals = re.findall(r"<suhu[^>]*>([^<]+)</suhu>", raw_str, re.I)
    suhu_vals += re.findall(r"<t[^a-z][^>]*>([^<]+)</t>", raw_str)
    suhu_vals += re.findall(r"<t_min[^>]*>([^<]+)</t_min>", raw_str)
    suhu_vals += re.findall(r"<t_max[^>]*>([^<]+)</t_max>", raw_str)
    cuaca_vals = re.findall(r"<cuaca[^>]*>([^<]+)</cuaca>", raw_str, re.I)
    kelembapan_vals = re.findall(r"<kelembaban[^>]*>([^<]+)</kelembaban>", raw_str, re.I)
    kelembapan_vals += re.findall(r"<humidity[^>]*>([^<]+)</humidity>", raw_str, re.I)
    if not suhu_vals and not cuaca_vals:
        return []
    # Build one row per period (prefer matching by index)
    n = max(len(suhu_vals), len(cuaca_vals), 1)
    for i in range(n):
        suhu_s = suhu_vals[i] if i < len(suhu_vals) else (suhu_vals[0] if suhu_vals else "25")
        cuaca_s = cuaca_vals[i] if i < len(cuaca_vals) else (cuaca_vals[0] if cuaca_vals else "Berawan")
        try:
            suhu = float(re.sub(r"[^\d.-]", "", suhu_s)) if suhu_s else 25.0
        except ValueError:
            suhu = 25.0
        # Kelembapan: BMKG gives %, store as 0–1 for consistency with model
        if kelembapan_vals and i < len(kelembapan_vals):
            try:
                k = float(re.sub(r"[^\d.]", "", kelembapan_vals[i]))
                kelembapan = k / 100.0 if k > 1 else k
            except ValueError:
                kelembapan = 0.7
        else:
            kelembapan = 0.7
        rows.append({
            "suhu": suhu,
            "cuaca": _cuaca_to_code(cuaca_s),
            "kelembapan": kelembapan,
            "ph": 6.0,  # BMKG does not provide pH; keep default for CSV, replace with sensor later
        })
    return rows


def parse_json_cuaca_suhu(raw):
    """If BMKG returns JSON, parse to list of {suhu, cuaca, kelembapan}."""
    import json
    try:
        data = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return []
    rows = []
    # Flexible: accept array of objects or nested .data / .prakiraan
    if isinstance(data, list):
        arr = data
    elif isinstance(data, dict):
        arr = data.get("data") or data.get("prakiraan") or []
        if isinstance(arr, dict):
            arr = [arr]
    else:
        return []
    for item in arr:
        if not isinstance(item, dict):
            continue
        # Nested prakiraan.sekarang / besok
        prak = item.get("prakiraan") or item
        for key in ("sekarang", "besok", "prakiraan"):
            p = prak if key == "prakiraan" and isinstance(prak, dict) else prak.get(key)
            if not isinstance(p, dict):
                continue
            cuaca_s = p.get("cuaca") or p.get("kode_cuaca") or "Berawan"
            suhu = p.get("suhu")
            if isinstance(suhu, dict):
                min_t = float(suhu.get("min") or suhu.get("t_min") or 22)
                max_t = float(suhu.get("max") or suhu.get("t_max") or 28)
                suhu = (min_t + max_t) / 2
            else:
                suhu = float(suhu) if suhu is not None else 25.0
            kelembaban = p.get("kelembaban") or p.get("kelembapan") or p.get("humidity") or {}
            if isinstance(kelembaban, dict):
                h = kelembaban.get("max") or kelembaban.get("min") or 80
                kelembapan = float(h) / 100.0 if float(h) > 1 else float(h)
            else:
                kelembapan = float(kelembaban) / 100.0 if float(kelembaban) > 1 else float(kelembaban)
            rows.append({
                "suhu": suhu,
                "cuaca": _cuaca_to_code(str(cuaca_s)),
                "kelembapan": kelembapan,
                "ph": 6.0,
            })
    if not rows and arr:
        one = arr[0] if isinstance(arr[0], dict) else {}
        suhu = one.get("suhu", 25)
        if isinstance(suhu, dict):
            suhu = (float(suhu.get("min", 22)) + float(suhu.get("max", 28))) / 2
        rows.append({
            "suhu": float(suhu),
            "cuaca": _cuaca_to_code(str(one.get("cuaca", "Berawan"))),
            "kelembapan": 0.7,
            "ph": 6.0,
        })
    return rows


def append_to_csv(rows, path="data/realtime_data.csv"):
    """Append rows to CSV; create file with header if missing."""
    if not rows:
        return
    filepath = path
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["suhu", "cuaca", "kelembapan", "ph"])
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"Appended {len(rows)} row(s) to {filepath}")


def fetch_bmkg_rows(save_to_bmkg_csv=True):
    """
    Fetch BMKG weather/temperature for Coblong. Returns list of dicts with
    suhu, cuaca, kelembapan, ph (ph=6.0 default). Optionally saves to
    data/bmkg_coblong.csv for use by build_internet_data.py.
    """
    raw = fetch_bmkg_xml(API_URL)
    if not raw:
        return []
    rows = parse_json_cuaca_suhu(raw)
    if not rows:
        rows = parse_xml_cuaca_suhu(raw)
    if not rows:
        return []
    if save_to_bmkg_csv and rows:
        os.makedirs(os.path.dirname("data/bmkg_coblong.csv") or ".", exist_ok=True)
        with open("data/bmkg_coblong.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["suhu", "cuaca", "kelembapan", "ph"])
            w.writeheader()
            w.writerows(rows)
    return rows


def main():
    print(f"Fetching BMKG prakiraan cuaca for Coblong (adm4={BMKG_COBLONG_CODE})...")
    rows = fetch_bmkg_rows(save_to_bmkg_csv=True)
    if not rows:
        print("No data. Check data/cuaca-suhu for API and try adm4=32.73.02.1001 for a kelurahan.")
        return
    print(f"Fetched {len(rows)} row(s). Saved to data/bmkg_coblong.csv for build_internet_data.py.")


if __name__ == "__main__":
    main()

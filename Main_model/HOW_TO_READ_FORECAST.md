# How to Read prediction_forecast.csv

A plain-language guide to interpreting the numeric forecast.

---

## Column by column

| Column | What it is | How to read it | Good for lettuce? |
|--------|------------|----------------|-------------------|
| **time** | Date and hour of the prediction | `2026-02-21 08:00` = Feb 21, 2026, 8:00 AM | — |
| **suhu** | Air temperature (°C) | 20 = mild, 25 = warm, 30+ = hot | 15–25°C ideal; avoid extremes |
| **cuaca** | Weather code | **0** = clear, **1** = cloudy, **2** = rain | Clear (0) good for growth; rain (2) may need drainage |
| **kelembapan** | Soil moisture (fraction 0–1) | 0.36 = 36%, 0.55 = 55% | **0.35–0.65** good; **0.55–0.60** ideal |
| **ph** | Soil pH | 6.0–7.0 = neutral, good for most crops | **6.0–7.0** ideal; **~6.5** best for lettuce |

---

## Quick examples from your data

| Row | Interpretation |
|-----|----------------|
| `suhu=19.7, cuaca=0, kelembapan=0.36, ph=6.60` | Cool clear morning (19.7°C), dry soil (36%), pH OK (6.6) |
| `suhu=25.0, cuaca=0, kelembapan=0.35, ph=6.60` | Warm afternoon (25°C), soil a bit dry, pH OK |
| `suhu=20.0, cuaca=2, kelembapan=0.50, ph=6.55` | Rainy (cuaca=2), moisture rising, pH OK |

---

## For red leaf lettuce (your crop)

- **Temperature:** Best around 15–25°C; watch for cold (<10°C) or heat (>30°C)
- **Soil moisture:** Keep between **0.35–0.65** (35–65%); **0.55–0.60** is ideal
- **pH:** Keep between **6.0–7.0**; aim for ~6.5
- **Weather:** Clear (0) is fine; rain (2) may need irrigation or drainage checks

---

## Converting numbers mentally

- **kelembapan 0.36** → "36% soil moisture" → slightly dry
- **kelembapan 0.55** → "55% soil moisture" → ideal
- **ph 6.6** → "pH 6.6" → good for lettuce
- **cuaca 0** → "Clear sky"
- **cuaca 1** → "Cloudy"
- **cuaca 2** → "Rain"

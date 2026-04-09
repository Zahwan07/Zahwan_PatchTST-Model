# How to Read prediction_forecast.csv

Guide for **hydroponic red leaf lettuce**. Columns: Temperature, Humidity, Weather, Light intensity, pH (no soil moisture).

---

## Column by column

| Column | What it is | How to read it | Good for lettuce? |
|--------|------------|----------------|-------------------|
| **time** | Date and hour of the prediction | `2026-02-21 08:00` = Feb 21, 2026, 8:00 AM | — |
| **suhu** | Air temperature (°C) | 20 = mild, 25 = warm, 30+ = hot | **18–24°C** ideal |
| **cuaca** | Weather code | **0** = clear, **1** = cloudy, **2** = rain | Clear (0) good for light |
| **humidity** | Air humidity (0–1) | 0.55 = 55% | **0.50–0.70** (50–70%) ideal |
| **light_intensity** | Light (W/m²) | 0 = night, 200–500 = good, 800+ = strong sun | **150–600** W/m²; 6–8h sun or 12–16h LED |
| **ph** | Hydroponic solution pH | 6.0–7.0 = good | **6.0–7.0** ideal; ~6.5 best |

---

## Quick examples

| Row | Interpretation |
|-----|----------------|
| `suhu=20, cuaca=0, humidity=0.55, light_intensity=350, ph=6.5` | Cool clear day, good humidity and light, pH OK |
| `suhu=25, cuaca=1, humidity=0.65, light_intensity=200, ph=6.4` | Warm, cloudy, higher humidity, lower light |
| `suhu=22, cuaca=2, humidity=0.70, light_intensity=50, ph=6.6` | Rainy, high humidity, low light |

---

## For hydroponic red leaf lettuce

- **Temperature:** 18–24°C ideal
- **Humidity:** 50–70% (0.50–0.70)
- **Light:** 6–8 hours sunlight or 12–16 hours LED; ~150–600 W/m²
- **pH:** 6.0–7.0 (nutrient solution)
- **Weather:** Clear (0) good; rain (2) may affect humidity/light

---

## Converting numbers

- **humidity 0.55** → "55% air humidity"
- **light_intensity 350** → "350 W/m²" (moderate light)
- **ph 6.6** → good for lettuce
- **cuaca 0** = Clear, **1** = Cloudy, **2** = Rain

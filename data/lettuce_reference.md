# Red leaf lettuce (Lactuca sativa L.) — optimal soil moisture & pH

Reference values for **PatchTST plant growth prediction** when the target crop is **red leaf lettuce** (*Lactuca sativa* L.). Use these ranges to build or validate your dataset for soil moisture and pH until you have sensor data.

---

## Soil moisture (kelembapan)

- **Optimal range (field capacity basis):** **55–60%** of field water holding capacity.  
  This mild range can support good photosynthesis and plant performance.
- **Volumetric water content (sensor-style):** **0.30–0.40 m³/m³** (30–40% volumetric) is reported in irrigation studies for lettuce, with good water use efficiency and quality.
- **In this project:** Store as a fraction **0–1** (e.g. 0.55–0.60 for “optimal”, or 0.35–0.40 if using VWC).  
  **Suggested optimal band:** **0.35–0.65**; **ideal center:** **0.55–0.60**.

**Practical notes:**  
- Keep moisture **consistent** over the growth cycle to reduce tip-burn (water stress).  
- Avoid long periods of under-irrigation (yield loss) or over-irrigation (disease, leaching).

*References: e.g. vegetable irrigation – leafy greens (USU Extension); moisture sensor vs timer irrigation for soilless lettuce (HortSci); soil moisture temporal variance and water use efficiency in romaine lettuce.*

---

## Soil / substrate pH

- **Optimal range for lettuce:** **6.0–7.0** (slightly acidic to neutral).  
  Often cited as ideal for commercial lettuce production.
- **Wider suitable range:** **5.5–7.0** for most vegetables, including lettuce; outside this range nutrient availability and toxicity risks increase (e.g. Fe/Mn at high pH, Al/Fe/Mn/Zn at low pH).
- **In this project:** Use pH values in **6.0–7.0** as “optimal” for *Lactuca sativa*; **ideal center around 6.5**.

*References: e.g. soil pH for vegetable production (UF/IFAS EDIS); lettuce soil requirements (Agrownet); NC State / Clemson lettuce fact sheets.*

---

## Summary for model and dataset

| Parameter     | Column      | Optimal range | Ideal (center) | Unit in CSV |
|--------------|-------------|----------------|----------------|-------------|
| Soil moisture| `kelembapan`| 0.35–0.65     | 0.55–0.60      | 0–1 fraction |
| Soil/substrate pH | `ph` | 6.0–7.0       | ~6.5           | pH scale    |

Weather (`cuaca`) and temperature (`suhu`) are filled from BMKG (Coblong, Bandung). For training data, you can use the reference CSV or the generator script so that `kelembapan` and `ph` stay within these lettuce-optimal ranges until you have real sensor data.

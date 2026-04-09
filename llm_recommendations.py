"""
Plant recommendations via Qwen (Ollama) only. Hydroponic red leaf lettuce:
Temperature, Humidity, Weather, Light intensity, pH. No soil moisture.
"""
import os

# Weather code to text
CUACA_LABELS = {0: "cerah (clear)", 1: "berawan (cloudy)", 2: "hujan ringan (light rain)", 3: "hujan sedang", 4: "hujan lebat", 5: "petir"}

# Always use local Ollama (Qwen) for recommendations
OLLAMA_BASE = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:14b"


def _get_client():
    """Return OpenAI-compatible client pointed at Ollama (Qwen)."""
    import openai
    client = openai.OpenAI(api_key="ollama", base_url=OLLAMA_BASE)
    return client


def forecast_to_text(forecast_7d):
    """
    forecast_7d: list of dicts or rows with suhu, cuaca, humidity, light_intensity, ph.
    """
    lines = []
    for i, row in enumerate(forecast_7d, 1):
        if isinstance(row, dict):
            suhu = row["suhu"]
            cuaca = row["cuaca"]
            humidity = row.get("humidity", row.get("kelembapan", 0))
            light = row.get("light_intensity", 0)
            ph = row["ph"]
        else:
            suhu, cuaca, humidity, light, ph = row[0], row[1], row[2], row[3], row[4]
        cuaca_str = CUACA_LABELS.get(int(round(cuaca)), "berawan")
        lines.append(
            f"Day {i}: temperature {suhu:.1f}°C, weather {cuaca_str}, "
            f"humidity {humidity:.2f} (0–1), light intensity {light:.0f} W/m², pH {ph:.2f}."
        )
    return "\n".join(lines)


def get_llm_recommendations(forecast_text, crop="red leaf lettuce (Lactuca sativa L., hydroponic)", location="Bandung (Coblong)"):
    """Get recommendations from Qwen (Ollama) only."""
    prompt = f"""You are an advisor for hydroponic lettuce growers. Write short, plain recommendations. Use simple everyday words.

Rules:
- Start with the current situation: e.g. "Current plant condition is optimal." OR "Current temperature is 0.5 degrees higher than optimal." OR "Current humidity is 0.48."
- When something needs adjustment, give ONE specific number: e.g. "Adjust light intensity to 400 W/m²" or "Adjust pH to 6.5". Do NOT say "between X and Y" or "150–600 W/m²".
- If the summary mentions predicted conditions, add one line like: "Monitor conditions for the next [blank hours] when temperatures are predicted to peak at [blank temperature]°C."
- Keep it short (2–4 sentences). No bullet points. No technical jargon. No soil or irrigation.

Forecast:
{forecast_text}
"""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", OLLAMA_MODEL),
            messages=[
                {"role": "system", "content": "Reply only with the recommendation in plain, simple language. One short paragraph."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return "Qwen returned no text. Is Ollama running? Try: ollama run qwen2.5:14b"
        return text
    except Exception as e:
        return f"Qwen request failed: {e}\n\nIs Ollama running? Start it and run: ollama run qwen2.5:14b"


def get_recommendations_from_forecast(forecast_7d, crop="red leaf lettuce (Lactuca sativa L., hydroponic)", location="Bandung (Coblong)"):
    """Build forecast text and get Qwen recommendations."""
    text = forecast_to_text(forecast_7d)
    return get_llm_recommendations(text, crop=crop, location=location)


def get_recommendations_from_current_conditions(
    today_summary,
    crop="red leaf lettuce (Lactuca sativa L., hydroponic)",
    location="Bandung (Coblong)",
    current_optimal=False,
    optimal_hours=0,
):
    """
    Send current conditions to Qwen. If current_optimal and optimal_hours >= 1,
    prompt can yield: "Plant is currently in optimal shape. Please check again in N hours."
    """
    instruction = ""
    if current_optimal and optimal_hours >= 1:
        instruction = (
            f"If the summary says current conditions are optimal and predicted optimal for the next {optimal_hours} hours, "
            f"reply with only: \"Plant is currently in optimal shape. Please check again in {optimal_hours} hours for any issues.\" "
            "Otherwise give normal recommendations.\n\n"
        )
    prompt = f"""You are an advisor for hydroponic lettuce. Write short, plain recommendations. Use simple everyday words.

Rules:
- Say the current situation clearly: e.g. "Current plant condition is optimal." OR "Current temperature is 0.2 degrees higher than optimal." OR "Current humidity is 0.48, please adjust."
- When suggesting changes, give ONE number only: e.g. "Adjust light intensity to 400 W/m²" or "Adjust pH to 6.5". Never use ranges like "between 150-600" or "0.50–0.70".
- If the summary mentions predicted conditions, add one line like: "Monitor conditions for the next [blank hours] when temperatures are predicted to peak at [blank temperature]°C."
- Keep it short (2–4 sentences). No bullet points. No technical terms. No soil or irrigation.

{instruction}Current conditions and optimal ranges:
{today_summary}
"""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", OLLAMA_MODEL),
            messages=[
                {"role": "system", "content": "Reply only with the recommendation in plain, simple language. One short paragraph."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return "Qwen returned no text. Is Ollama running? Try: ollama run qwen2.5:14b"
        return text
    except Exception as e:
        return f"Qwen request failed: {e}\n\nIs Ollama running? Run: ollama run qwen2.5:14b"

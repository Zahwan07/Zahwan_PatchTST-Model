"""
Send the 7-day environment forecast (numeric) to an LLM API and get
natural-language recommendations for the user (e.g. red leaf lettuce in Bandung).

Set OPENAI_API_KEY in the environment, or LLM_API_KEY and LLM_API_BASE for a compatible API.
"""
import os
import json

# Weather code to text for the prompt
CUACA_LABELS = {0: "cerah (clear)", 1: "berawan (cloudy)", 2: "hujan ringan (light rain)", 3: "hujan sedang", 4: "hujan lebat", 5: "petir"}


def forecast_to_text(forecast_7d):
    """
    forecast_7d: list of 7 dicts with keys suhu, cuaca, kelembapan, ph
    or (7, 4) array in order [suhu, cuaca, kelembapan, ph].
    """
    lines = []
    for i, row in enumerate(forecast_7d, 1):
        if isinstance(row, dict):
            suhu, cuaca, kelembapan, ph = row["suhu"], row["cuaca"], row["kelembapan"], row["ph"]
        else:
            suhu, cuaca, kelembapan, ph = row[0], row[1], row[2], row[3]
        cuaca_str = CUACA_LABELS.get(int(round(cuaca)), "berawan")
        lines.append(
            f"Day {i}: temperature {suhu:.1f}°C, weather {cuaca_str}, "
            f"soil moisture {kelembapan:.2f} (0-1), soil pH {ph:.2f}."
        )
    return "\n".join(lines)


def get_llm_recommendations(forecast_text, crop="red leaf lettuce (Lactuca sativa L.)", location="Bandung (Coblong)"):
    """
    Call LLM API with the forecast text and return recommendation string.
    Uses OPENAI_API_KEY and OpenAI API by default; optional LLM_API_BASE for compatible endpoints.
    """
    prompt = f"""You are an agricultural advisor for vegetable growers. Given the following 7-day environment forecast for {crop} in {location}, write brief recommendations for the user in 2–4 short paragraphs. Use plain language.

Focus on:
- What to expect (e.g. "The weather will likely stay around X°C for the next few days.")
- Soil moisture and irrigation (e.g. "Expect soil moisture to decrease; consider watering.")
- Soil pH if relevant.
- Any practical tips (shading, covering, watering schedule).

Do not use bullet points; write flowing sentences. Write in the same language as this message (English).

7-day forecast:
{forecast_text}
"""

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        return (
            "LLM recommendations skipped: set OPENAI_API_KEY (or LLM_API_KEY) in the environment to get AI-generated advice.\n"
            "Here is the numeric forecast above for your reference."
        )

    try:
        import openai
    except ImportError:
        return (
            "LLM recommendations skipped: install openai (pip install openai) and set OPENAI_API_KEY.\n"
            "Here is the numeric forecast above for your reference."
        )

    client = openai.OpenAI(api_key=api_key)
    base = os.environ.get("LLM_API_BASE")
    if base:
        client.base_url = base

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a concise agricultural advisor. Reply only with the recommendations, no preamble."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"LLM request failed: {e}\nUse the numeric forecast above for your reference."


def get_recommendations_from_forecast(forecast_7d, crop="red leaf lettuce (Lactuca sativa L.)", location="Bandung (Coblong)"):
    """Build forecast text from 7-day data and return LLM recommendations."""
    text = forecast_to_text(forecast_7d)
    return get_llm_recommendations(text, crop=crop, location=location)

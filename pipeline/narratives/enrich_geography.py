"""Generate 'why this region' geographic context for each hub.

Reads:
  pipeline/clustered/hubs.json
  pipeline/climate/climate.json
  pipeline/narratives/hubs.json (existing narratives)

Writes:
  pipeline/narratives/hubs.json (in-place update with new geographic_context field)

Uses Vertex AI Gemini 2.5 Flash. Each call is grounded in real climate data
plus the hub's actual top sports, so output cites specific numbers.

Note on phrasing: the contest brief says use conditional phrasing like 'could
help find', not deterministic claims. Prompt enforces this.
"""
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types as genai_types

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HUBS_PATH = PROJECT_ROOT / "pipeline" / "clustered" / "hubs.json"
CLIMATE_PATH = PROJECT_ROOT / "pipeline" / "climate" / "climate.json"
NARRATIVES_PATH = PROJECT_ROOT / "pipeline" / "narratives" / "hubs.json"


SYSTEM_PROMPT = """You are an expert geographer and sports analyst writing
context cards for an interactive map called the Hometown Success Engine.

Your job: for each US hometown hub, write a 2-sentence "why this region"
paragraph that describes how the local geography, climate, terrain, or
infrastructure could support the development of athletes in the sports that
actually appear in this hub.

CRITICAL RULES:
- Use CONDITIONAL phrasing only. NEVER claim geography "produces" or "creates"
  athletes. Use "could help foster", "is well-suited to", "supports access to",
  "may explain", "the conditions for".
- Cite the specific climate or geographic features given to you (temperature,
  precipitation, elevation, terrain).
- Tie the climate features to the actual top sports in this hub.
- Stay neutral and factual. No hype words like "incredible", "powerhouse", "elite".
- Output exactly 2 sentences. No more, no less. No headers, no bullet points.
- Do not name specific athletes, teams, or NGB organizations.
- Do not use em dashes (—). Use commas or periods.
"""


USER_TEMPLATE = """Hub: {display_name} ({region_name}, {macro_region})
Top sports in this hub (with athlete count): {top_sports}
Annual avg temperature: {temp_f}°F
Annual precipitation: {precip_in} inches
Annual sunshine: {sunshine_hours} hours
Elevation: {elevation_ft} feet
Paralympic share: {para_pct}% (national baseline 4.6%)
{hot_spot_note}

Write the 2-sentence geographic context."""


def build_user_prompt(hub: dict, climate: dict) -> str:
    top_sports_str = ", ".join(
        f"{s['sport']} ({s['count']} athletes)"
        for s in hub.get("top_sports", [])[:4]
    ) or "various sports"

    para_pct = round(hub["composition"]["paralympic_share"] * 100, 1)
    hot_spot_note = (
        "This is a Paralympic Hot Spot (Para share runs more than 2x national rate)."
        if hub.get("is_paralympic_hot_spot") else ""
    )

    return USER_TEMPLATE.format(
        display_name=hub["display_name"],
        region_name=hub.get("region_name", ""),
        macro_region=hub.get("macro_region", ""),
        top_sports=top_sports_str,
        temp_f=climate.get("annual_avg_temp_f", "unknown"),
        precip_in=climate.get("annual_precipitation_in", "unknown"),
        sunshine_hours=climate.get("annual_sunshine_hours", "unknown"),
        elevation_ft=climate.get("elevation_ft", "unknown"),
        para_pct=para_pct,
        hot_spot_note=hot_spot_note,
    )


def main():
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "hometown-success-engine")
    client = genai.Client(vertexai=True, project=project, location="global")

    hubs = json.loads(HUBS_PATH.read_text(encoding="utf-8"))
    climate_all = json.loads(CLIMATE_PATH.read_text(encoding="utf-8"))
    narratives = json.loads(NARRATIVES_PATH.read_text(encoding="utf-8"))

    print(f"Enriching {len(hubs)} hubs with geographic_context...")

    updated = 0
    for i, hub in enumerate(hubs):
        hub_id = hub["hub_id"]
        climate = climate_all.get(hub_id) or {}
        if not climate:
            print(f"  [{i+1}/{len(hubs)}] {hub_id}... SKIP (no climate)")
            continue

        prompt = build_user_prompt(hub, climate)
        print(f"  [{i+1}/{len(hubs)}] {hub_id}... ", end="", flush=True)

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])],
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.5,
                    max_output_tokens=800,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = ""
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text

            text = text.strip().replace("\u2014", ",").replace("—", ",")

            if hub_id not in narratives:
                print(f"SKIP (no narrative entry)")
                continue

            narratives[hub_id]["geographic_context"] = text
            narratives[hub_id]["climate"] = {
                "annual_avg_temp_f": climate.get("annual_avg_temp_f"),
                "annual_precipitation_in": climate.get("annual_precipitation_in"),
                "annual_sunshine_hours": climate.get("annual_sunshine_hours"),
                "elevation_ft": climate.get("elevation_ft"),
            }
            updated += 1
            print(f"OK ({len(text)} chars)")
        except Exception as e:
            print(f"FAIL: {e}")

        time.sleep(0.3)

    NARRATIVES_PATH.write_text(json.dumps(narratives, indent=2), encoding="utf-8")
    print(f"\nUpdated {updated}/{len(hubs)} narratives in {NARRATIVES_PATH}")


if __name__ == "__main__":
    main()
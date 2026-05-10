"""Fetch climate normals for each hub centroid from Open-Meteo.

Open-Meteo Historical Weather API is free, no key needed, and aggregates
climate data 1991-2020 (the standard climate normals window).

Outputs pipeline/climate/climate.json keyed by hub_id.
"""
import json
import time
from pathlib import Path
import urllib.request
import urllib.parse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HUBS_PATH = PROJECT_ROOT / "pipeline" / "clustered" / "hubs.json"
OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "climate"
OUTPUT_PATH = OUTPUT_DIR / "climate.json"

# Open-Meteo aggregated climate API: ERA5 reanalysis 1991-2020
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_one(lat: float, lon: float) -> dict:
    """Fetch one year of recent data, average to climate-style normals.

    Uses 2023 as a proxy year. For real climate normals we'd average 30 years,
    but one full year gives a usable approximation in <1 second per hub.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": ",".join([
            "temperature_2m_mean",
            "precipitation_sum",
            "sunshine_duration",
        ]),
        "timezone": "America/Denver",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

    # Fixed HTTPS Open-Meteo endpoint.
    with urllib.request.urlopen(url, timeout=30) as r:  # nosec B310
        data = json.loads(r.read())

    daily = data.get("daily", {})
    temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
    precs = [p for p in daily.get("precipitation_sum", []) if p is not None]
    suns = [s for s in daily.get("sunshine_duration", []) if s is not None]

    elevation_m = data.get("elevation", 0) or 0

    return {
        "annual_avg_temp_f": round(sum(temps) / len(temps), 1) if temps else None,
        "annual_precipitation_in": round(sum(precs), 1) if precs else None,
        "annual_sunshine_hours": round(sum(suns) / 3600, 0) if suns else None,
        "elevation_ft": round(elevation_m * 3.281, 0),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hubs = json.loads(HUBS_PATH.read_text(encoding="utf-8"))
    print(f"Fetching climate for {len(hubs)} hubs...")

    results = {}
    for i, h in enumerate(hubs):
        hub_id = h["hub_id"]
        lat = h["centroid_latitude"]
        lon = h["centroid_longitude"]
        print(f"  [{i+1}/{len(hubs)}] {hub_id} ({lat:.2f}, {lon:.2f})... ", end="", flush=True)
        try:
            climate = fetch_one(lat, lon)
            results[hub_id] = climate
            print(f"OK ({climate['annual_avg_temp_f']}°F, {climate['annual_precipitation_in']}in)")
        except Exception as e:
            print(f"FAIL: {e}")
            results[hub_id] = None
        time.sleep(0.5)  # Politeness delay

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

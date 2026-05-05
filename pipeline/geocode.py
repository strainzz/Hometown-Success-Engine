import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

import httpx

from pipeline.normalize import Athlete


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_USER_AGENT = (
    "HometownSuccessEngine/0.1.0 "
    "(https://github.com/strainzz/Hometown-Success-Engine; "
    "strainz@galluslabs.com; "
    "Hackathon submission, not for commercial use)"
)


def parse_wkt_point(wkt: str) -> tuple[float, float] | None:
    """Parses 'Point(LONG LAT)' format. Returns (lat, long) tuple
    matching the Hometown model's field order. Returns None on parse
    failure."""
    if not wkt.startswith("Point(") or not wkt.endswith(")"):
        return None
    inner = wkt[6:-1]
    parts = inner.split()
    if len(parts) != 2:
        return None
    try:
        lon = float(parts[0])
        lat = float(parts[1])
        return (lat, lon)
    except ValueError:
        return None


async def fetch_coordinates_batch(
    place_uris: list[str],
    client: httpx.AsyncClient
) -> dict[str, tuple[float, float]]:
    """Returns {wikidata_uri: (latitude, longitude)} for places that
    have wdt:P625 coordinates in Wikidata. Missing places simply absent
    from the dict."""
    uris_formatted = " ".join(f"<{uri}>" for uri in place_uris)
    query = f"""
    SELECT ?place ?coords WHERE {{
      VALUES ?place {{ {uris_formatted} }}
      ?place wdt:P625 ?coords.
    }}
    """
    
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/sparql-results+json"
    }
    
    delays = [2, 8, 30]
    max_attempts = len(delays) + 1
    
    for attempt in range(max_attempts):
        try:
            response = await client.post(
                DEFAULT_ENDPOINT,
                data={"query": query},
                headers=headers
            )
            
            if response.status_code in (429, 503):
                if attempt < len(delays):
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        sleep_time = float(retry_after)
                    else:
                        sleep_time = delays[attempt]
                    logger.warning(f"HTTP {response.status_code}. Retrying in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    response.raise_for_status()
                    
            response.raise_for_status()
            data = response.json()
            
            results = {}
            for row in data.get("results", {}).get("bindings", []):
                place_uri = row.get("place", {}).get("value")
                coords_wkt = row.get("coords", {}).get("value")
                if place_uri and coords_wkt:
                    parsed = parse_wkt_point(coords_wkt)
                    if parsed:
                        results[place_uri] = parsed
            return results
            
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < len(delays):
                logger.warning(f"Request failed: {e}. Retrying in {delays[attempt]}s...")
                await asyncio.sleep(delays[attempt])
            else:
                logger.error(f"Failed batch after {max_attempts} attempts: {e}")
                return {}
    
    return {}


async def geocode_athletes(
    athletes: list[Athlete]
) -> tuple[list[Athlete], dict[str, int]]:
    """Geocodes all unique hometown URIs from the athletes list, then
    populates each athlete's hometown with lat/long where resolved.
    Returns (geocoded_athletes, stats_dict)."""
    unique_uris = {a.hometown.wikidata_uri for a in athletes}
    uri_list = list(unique_uris)
    
    batch_size = 500
    batches = [uri_list[i:i + batch_size] for i in range(0, len(uri_list), batch_size)]
    
    coords_map = {}
    
    async with httpx.AsyncClient(timeout=50.0) as client:
        for idx, batch in enumerate(batches, 1):
            batch_coords = await fetch_coordinates_batch(batch, client)
            coords_map.update(batch_coords)
            logger.info(f"Batch {idx}/{len(batches)} returned {len(batch_coords)} results")
            
            if idx < len(batches):
                await asyncio.sleep(2)
                
    resolved = 0
    for athlete in athletes:
        uri = athlete.hometown.wikidata_uri
        if uri in coords_map:
            lat, lon = coords_map[uri]
            athlete.hometown.latitude = lat
            athlete.hometown.longitude = lon
            resolved += 1
            
    stats = {
        "unique_hometowns": len(unique_uris),
        "resolved": len(coords_map),
        "missing": len(unique_uris) - len(coords_map),
        "batches_run": len(batches)
    }
    
    return athletes, stats


async def main() -> None:
    base_dir = Path("pipeline")
    in_path = base_dir / "normalized" / "athletes.json"
    out_path = base_dir / "geocoded" / "athletes.json"
    
    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        return

    logger.info(f"Loading normalized athletes from {in_path}...")
    with in_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    athletes = [Athlete.model_validate(item) for item in raw_data]
    
    logger.info(f"Loaded {len(athletes)} athletes. Starting geocoding process...")
    geocoded_athletes, stats = await geocode_athletes(athletes)
    
    missing_hometowns_counter: Counter[str] = Counter()
    for a in geocoded_athletes:
        if a.hometown.latitude is None or a.hometown.longitude is None:
            missing_hometowns_counter[a.hometown.label] += 1
            
    coverage = (stats['resolved'] / stats['unique_hometowns'] * 100) if stats['unique_hometowns'] > 0 else 0.0
    
    logger.info("--- GEOCODING SUMMARY ---")
    logger.info(f"Number of unique hometowns to geocode: {stats['unique_hometowns']}")
    logger.info(f"Total resolved: {stats['resolved']}")
    logger.info(f"Total missing: {stats['missing']}")
    logger.info(f"Coverage percentage: {coverage:.1f}%")
    
    logger.info("Top 10 missing hometowns by athlete count:")
    for ht_label, count in missing_hometowns_counter.most_common(10):
        logger.info(f"  {ht_label}: {count} athletes")
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_data = [a.model_dump(mode="json") for a in geocoded_athletes]
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully wrote geocoded athletes to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
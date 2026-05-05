import asyncio
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_USER_AGENT = "HometownSuccessEngine/0.1.0 (https://github.com/strainzz/Hometown-Success-Engine; strainz@galluslabs.com; Hackathon submission, not for commercial use)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OLYMPIC_QUERY = """
SELECT DISTINCT ?athlete ?athleteLabel ?sportLabel ?birthPlace ?birthPlaceLabel
    (GROUP_CONCAT(DISTINCT ?medalLabel; SEPARATOR=", ") AS ?medals)
    ?genderLabel ?dateOfBirth
WHERE {
  ?athlete wdt:P8286 ?olympediaId.
  ?athlete wdt:P31 wd:Q5.
  { ?athlete wdt:P27 wd:Q30. } UNION { ?athlete wdt:P1532 wd:Q30. }
  ?athlete wdt:P641 ?sport.
  ?athlete wdt:P19 ?birthPlace.
  ?birthPlace wdt:P17 wd:Q30.
  OPTIONAL {
    ?athlete wdt:P166 ?medal.
    VALUES ?medal { wd:Q15243387 wd:Q15889641 wd:Q15889643 }
    ?medal rdfs:label ?medalLabel.
    FILTER(LANG(?medalLabel) = "en")
  }
  OPTIONAL { ?athlete wdt:P21 ?gender. }
  OPTIONAL { ?athlete wdt:P569 ?dateOfBirth. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?athlete ?athleteLabel ?sportLabel ?birthPlace ?birthPlaceLabel ?genderLabel ?dateOfBirth
LIMIT 5000
"""

PARALYMPIC_QUERY = """
SELECT DISTINCT ?athlete ?athleteLabel ?sportLabel ?birthPlace ?birthPlaceLabel
    (GROUP_CONCAT(DISTINCT ?medalLabel; SEPARATOR=", ") AS ?medals)
    ?genderLabel ?dateOfBirth
WHERE {
  ?athlete wdt:P7550 ?paraId.
  ?athlete wdt:P31 wd:Q5.
  { ?athlete wdt:P27 wd:Q30. } UNION { ?athlete wdt:P1532 wd:Q30. }
  ?athlete wdt:P641 ?sport.
  ?athlete wdt:P19 ?birthPlace.
  ?birthPlace wdt:P17 wd:Q30.
  OPTIONAL {
    ?athlete wdt:P166 ?medal.
    ?medal wdt:P279* wd:Q15243428.
    ?medal rdfs:label ?medalLabel.
    FILTER(LANG(?medalLabel) = "en")
  }
  OPTIONAL { ?athlete wdt:P21 ?gender. }
  OPTIONAL { ?athlete wdt:P569 ?dateOfBirth. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?athlete ?athleteLabel ?sportLabel ?birthPlace ?birthPlaceLabel ?genderLabel ?dateOfBirth
LIMIT 5000
"""

async def execute_sparql_query(query: str, client: httpx.AsyncClient) -> dict[str, Any]:
    endpoint = os.environ.get("WIKIDATA_SPARQL_ENDPOINT", DEFAULT_ENDPOINT)
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": os.environ.get("USER_AGENT", DEFAULT_USER_AGENT),
    }
    delays = [2, 8, 30]

    for attempt in range(4):
        try:
            response = await client.post(
                endpoint,
                data={"query": query},
                headers=headers
            )
            
            if response.status_code in (429, 503):
                if attempt < 3:
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = int(retry_after) if retry_after and retry_after.isdigit() else delays[attempt]
                    logger.warning(f"Rate limited (HTTP {response.status_code}). Retrying in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    response.raise_for_status()
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            if attempt < 3:
                logger.warning(f"Request error: {e}. Retrying in {delays[attempt]}s...")
                await asyncio.sleep(delays[attempt])
            else:
                logger.error("Max retries exceeded.")
                raise

    raise RuntimeError("Failed to fetch data from Wikidata after retries.")

def log_summary(data: dict[str, Any], dataset_name: str) -> None:
    bindings = data.get("results", {}).get("bindings", [])
    total_athletes = len(bindings)
    
    sport_counts: Counter[str] = Counter()
    hometown_count = 0
    medal_count = 0
    
    for item in bindings:
        sport = item.get("sportLabel", {}).get("value")
        if sport:
            sport_counts[sport] += 1
            
        birth_place = item.get("birthPlace", {}).get("value")
        if birth_place:
            hometown_count += 1
            
        medals = item.get("medals", {}).get("value", "")
        if medals:
            medal_count += 1
            
    logger.info(f"--- {dataset_name} Summary ---")
    logger.info(f"Total Athletes: {total_athletes}")
    logger.info(f"Athletes with usable hometown: {hometown_count}")
    logger.info(f"Athletes with medal claims: {medal_count}")
    
    logger.info("Top 5 Sports:")
    for sport, count in sport_counts.most_common(5):
        logger.info(f"    {sport}: {count}")
    logger.info("-" * 30)

async def main() -> None:
    output_dir = Path("pipeline/raw_data/wikidata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    olympic_file = output_dir / "athletes_olympic.json"
    paralympic_file = output_dir / "athletes_paralympic.json"
    
    async with httpx.AsyncClient(timeout=50.0) as client:
        logger.info("Fetching Olympic athletes from Wikidata...")
        olympic_data = await execute_sparql_query(OLYMPIC_QUERY, client)
        
        with open(olympic_file, "w", encoding="utf-8") as f:
            json.dump(olympic_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Olympic athletes to {olympic_file}")
        log_summary(olympic_data, "Olympic Athletes")
        
        logger.info("Sleeping for 5 seconds to respect endpoint rate limits...")
        await asyncio.sleep(5)
        
        logger.info("Fetching Paralympic athletes from Wikidata...")
        paralympic_data = await execute_sparql_query(PARALYMPIC_QUERY, client)
        
        with open(paralympic_file, "w", encoding="utf-8") as f:
            json.dump(paralympic_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Paralympic athletes to {paralympic_file}")
        log_summary(paralympic_data, "Paralympic Athletes")

if __name__ == "__main__":
    asyncio.run(main())
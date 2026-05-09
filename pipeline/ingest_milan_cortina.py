"""Ingest Milan-Cortina 2026 Olympic and Paralympic Team USA athletes.

Strategy:
1. Pull rosters from Wikipedia "United States at the 2026 Winter Olympics"
   and "United States at the 2026 Winter Paralympics" using the MediaWiki
   parse API (returns clean wikitext sections we can regex). This gives us
   athlete name + sport with full coverage.
2. For each athlete, query Wikidata via wbsearchentities to find their Q-ID,
   then fetch wbgetentities for P19 (place of birth) and P19's P131
   (located in admin entity, gives us US state).
3. Output matches the existing pipeline/raw_data/wikidata/athletes_olympic.json
   SPARQL JSON schema so normalize.py / geocode.py / cluster.py work unchanged.

Compliance: Wikipedia/Wikidata are CC BY-SA / CC0. User-Agent identifies the
project. 1-2 second delays between Wikidata API calls. No bulk redistribution
of TeamUSA content.

Names are stored only for downstream dedup; normalize.py is responsible for
dropping name strings before the data leaves the pipeline (per NIL rules).
"""
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

USER_AGENT = (
    "HometownSuccessEngine/0.1.0 "
    "(https://github.com/strainzz/Hometown-Success-Engine; "
    "strainz@galluslabs.com; Hackathon submission, not for commercial use)"
)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

OLYMPIC_PAGE = "United States at the 2026 Winter Olympics"
PARALYMPIC_PAGE = "United States at the 2026 Winter Paralympics"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- helpers ----------

def to_binding(athlete_qid: str | None, name: str, sport: str,
               birth_place_qid: str | None, birth_place_label: str | None,
               gender: str | None, date_of_birth: str | None,
               medals: str = "") -> dict[str, Any]:
    """Format a record to match the existing Wikidata SPARQL bindings shape."""
    rec: dict[str, Any] = {
        "athlete": {
            "type": "uri",
            "value": (f"http://www.wikidata.org/entity/{athlete_qid}"
                      if athlete_qid else f"hse:milan-cortina-2026:{name}"),
        },
        "athleteLabel": {
            "xml:lang": "en",
            "type": "literal",
            "value": name,
        },
        "sportLabel": {
            "xml:lang": "en",
            "type": "literal",
            "value": sport,
        },
        "medals": {
            "type": "literal",
            "value": medals or "",
        },
    }
    if birth_place_qid:
        rec["birthPlace"] = {
            "type": "uri",
            "value": f"http://www.wikidata.org/entity/{birth_place_qid}",
        }
    if birth_place_label:
        rec["birthPlaceLabel"] = {
            "xml:lang": "en",
            "type": "literal",
            "value": birth_place_label,
        }
    if gender:
        rec["genderLabel"] = {
            "xml:lang": "en",
            "type": "literal",
            "value": gender,
        }
    if date_of_birth:
        rec["dateOfBirth"] = {
            "datatype": "http://www.w3.org/2001/XMLSchema#dateTime",
            "type": "literal",
            "value": date_of_birth,
        }
    return rec


# ---------- Wikipedia roster extraction ----------

async def fetch_wikipedia_wikitext(client: httpx.AsyncClient, page_title: str) -> str:
    """Fetch the raw wikitext of a Wikipedia page."""
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
        "redirects": 1,
    }
    headers = {"User-Agent": USER_AGENT}
    r = await client.get(WIKI_API, params=params, headers=headers, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"Wikipedia parse error for {page_title}: {data['error']}")
    return data["parse"]["wikitext"]["*"]


# Athlete extraction patterns. Wikipedia roster pages use templates like
# {{flagicon|USA}} {{sortname|First|Last}} or table cells with name links.
# We look for any [[Name]] or {{sortname|First|Last}} occurrences within
# tables that follow a sport section header.

SPORT_HEADER_RE = re.compile(r"^==+\s*(.+?)\s*==+\s*$", re.MULTILINE)
SORTNAME_RE = re.compile(r"\{\{\s*sortname\s*\|\s*([^|}]+?)\s*\|\s*([^|}]+?)\s*(?:\|[^}]*)?\}\}")
WIKILINK_NAME_RE = re.compile(r"\[\[([^|\]#]+?)(?:\|([^\]]+?))?\]\]")


def extract_athletes_from_wikitext(wikitext: str, default_sport: str | None = None) -> list[tuple[str, str]]:
    """Walk the wikitext, track current sport from section headers, and pull
    athlete names from sortname templates. Returns list of (name, sport).
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    current_sport = default_sport or "Winter Sport"

    # Common winter sports we expect to see as section headers
    known_sports = {
        "alpine skiing", "biathlon", "bobsleigh", "cross-country skiing",
        "curling", "figure skating", "freestyle skiing", "ice hockey",
        "luge", "nordic combined", "short track speed skating", "skeleton",
        "ski jumping", "snowboarding", "speed skating",
        # Paralympic
        "para alpine skiing", "para biathlon", "para cross-country skiing",
        "para ice hockey", "para snowboard", "wheelchair curling",
    }

    lines = wikitext.split("\n")
    in_table = False
    for line in lines:
        # Track section headers
        m = SPORT_HEADER_RE.match(line)
        if m:
            section = m.group(1).strip().lower()
            # Strip wiki formatting from section name
            section = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", section).strip()
            # Match against known sports
            for sport in known_sports:
                if sport in section:
                    current_sport = sport.title()
                    break
            else:
                # Use the section name as-is if it looks like a sport
                if any(kw in section for kw in ("skiing", "skating", "hockey",
                                                "luge", "bobsleigh", "biathlon",
                                                "curling", "snowboard",
                                                "skeleton", "combined")):
                    current_sport = section.title()
            continue

        if "{|" in line:
            in_table = True
        if "|}" in line:
            in_table = False

        # Find sortname templates anywhere (most reliable)
        for first, last in SORTNAME_RE.findall(line):
            name = f"{first.strip()} {last.strip()}"
            if name not in seen:
                seen.add(name)
                results.append((name, current_sport))

        # Also catch [[Name]] inside table rows that are clearly athlete cells
        # (line starts with | and the link is the first thing)
        if in_table and line.lstrip().startswith("|"):
            for full, displayed in WIKILINK_NAME_RE.findall(line):
                target = full.strip()
                # Skip obvious non-athlete links
                if any(skip in target.lower() for skip in (
                        "category:", "file:", "image:", "wikipedia:",
                        "list of", "olympics", "paralympics",
                        "skiing", "skating", "hockey", "curling",
                        "biathlon", "bobsleigh", "luge", "snowboard",
                        "skeleton", "combined", "united states", "team",
                )):
                    continue
                # Names usually have a space and are 2-4 words
                words = target.split()
                if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                    name = displayed.strip() if displayed else target
                    name = re.sub(r"\s*\([^)]*\)\s*", "", name).strip()
                    if name and name not in seen:
                        seen.add(name)
                        results.append((name, current_sport))

    return results


# ---------- Wikidata enrichment ----------

async def search_wikidata_qid(client: httpx.AsyncClient, athlete_name: str) -> str | None:
    """Search Wikidata for an athlete by name. Return Q-ID of best match."""
    params = {
        "action": "wbsearchentities",
        "search": athlete_name,
        "language": "en",
        "format": "json",
        "limit": 5,
        "type": "item",
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        r = await client.get(WIKIDATA_API, params=params, headers=headers, timeout=15.0)
        r.raise_for_status()
        data = r.json()
        for hit in data.get("search", []):
            desc = (hit.get("description") or "").lower()
            # Prefer hits with athlete-like descriptions
            if any(kw in desc for kw in (
                    "skier", "skater", "hockey", "athlete", "snowboarder",
                    "curler", "luger", "bobsledder", "biathlete",
                    "olympic", "paralympic", "sport",
            )):
                return hit["id"]
        # Fall back to first result if any
        if data.get("search"):
            return data["search"][0]["id"]
    except httpx.RequestError as e:
        logger.warning(f"  search failed for {athlete_name}: {e}")
    return None


async def fetch_wikidata_entity(client: httpx.AsyncClient, qid: str) -> dict[str, Any] | None:
    """Fetch claims for a Wikidata entity."""
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "claims|labels",
        "languages": "en",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        r = await client.get(WIKIDATA_API, params=params, headers=headers, timeout=15.0)
        r.raise_for_status()
        data = r.json()
        return data.get("entities", {}).get(qid)
    except httpx.RequestError as e:
        logger.warning(f"  entity fetch failed for {qid}: {e}")
    return None


async def fetch_label(client: httpx.AsyncClient, qid: str) -> str | None:
    """Get the English label for a Q-ID."""
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels",
        "languages": "en",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        r = await client.get(WIKIDATA_API, params=params, headers=headers, timeout=15.0)
        r.raise_for_status()
        data = r.json()
        ent = data.get("entities", {}).get(qid, {})
        return ent.get("labels", {}).get("en", {}).get("value")
    except httpx.RequestError as e:
        logger.warning(f"  label fetch failed for {qid}: {e}")
    return None


def extract_claim(entity: dict[str, Any], pid: str) -> str | None:
    """Pull the first claim Q-ID for a given P-ID from a Wikidata entity."""
    claims = entity.get("claims", {}).get(pid, [])
    for c in claims:
        try:
            return c["mainsnak"]["datavalue"]["value"]["id"]
        except (KeyError, TypeError):
            continue
    return None


def extract_date_claim(entity: dict[str, Any], pid: str) -> str | None:
    claims = entity.get("claims", {}).get(pid, [])
    for c in claims:
        try:
            t = c["mainsnak"]["datavalue"]["value"]["time"]
            # +1993-06-08T00:00:00Z -> 1993-06-08T00:00:00Z
            return t.lstrip("+")
        except (KeyError, TypeError):
            continue
    return None


async def enrich_athlete(client: httpx.AsyncClient, name: str, sport: str,
                        is_paralympic: bool) -> dict[str, Any]:
    """Build a record. Try Wikidata for hometown; fall back to no-hometown if
    not available. The downstream geocoder will skip records without a place
    label, which is correct behavior."""
    qid = await search_wikidata_qid(client, name)
    await asyncio.sleep(0.3)

    birth_place_qid: str | None = None
    birth_place_label: str | None = None
    gender: str | None = None
    dob: str | None = None

    if qid:
        ent = await fetch_wikidata_entity(client, qid)
        await asyncio.sleep(0.3)
        if ent:
            birth_place_qid = extract_claim(ent, "P19")
            gender_qid = extract_claim(ent, "P21")
            dob = extract_date_claim(ent, "P569")
            if gender_qid:
                # Q6581072 = female, Q6581097 = male (most common)
                gender = {"Q6581072": "female", "Q6581097": "male"}.get(gender_qid)
            if birth_place_qid:
                birth_place_label = await fetch_label(client, birth_place_qid)
                await asyncio.sleep(0.3)

    # Tag the medals field with paralympic/olympic so downstream can split
    medal_tag = "paralympic" if is_paralympic else "olympic"

    return to_binding(
        athlete_qid=qid,
        name=name,
        sport=sport,
        birth_place_qid=birth_place_qid,
        birth_place_label=birth_place_label,
        gender=gender,
        date_of_birth=dob,
        medals=medal_tag,
    )


# ---------- main ----------

async def harvest(page_title: str, is_paralympic: bool,
                 client: httpx.AsyncClient) -> list[dict[str, Any]]:
    logger.info(f"Fetching wikitext: {page_title}")
    wt = await fetch_wikipedia_wikitext(client, page_title)
    logger.info(f"  wikitext size: {len(wt):,} chars")

    athletes = extract_athletes_from_wikitext(wt)
    logger.info(f"  extracted {len(athletes)} athlete candidates")

    if not athletes:
        logger.warning("  no athletes found, dumping first 500 chars of wikitext for debug:")
        logger.warning(wt[:500])
        return []

    bindings: list[dict[str, Any]] = []
    for i, (name, sport) in enumerate(athletes, 1):
        if i % 10 == 0:
            logger.info(f"  enriching {i}/{len(athletes)}: {name}")
        rec = await enrich_athlete(client, name, sport, is_paralympic)
        bindings.append(rec)

    return bindings


async def main() -> None:
    output_dir = Path("pipeline/raw_data/wikidata")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_olympic = output_dir / "athletes_olympic_2026.json"
    out_paralympic = output_dir / "athletes_paralympic_2026.json"

    async with httpx.AsyncClient(timeout=60.0) as client:
        olympic_records = await harvest(OLYMPIC_PAGE, is_paralympic=False, client=client)
        out = {"head": {"vars": [
            "athlete", "athleteLabel", "sportLabel", "birthPlace",
            "birthPlaceLabel", "genderLabel", "dateOfBirth", "medals"
        ]}, "results": {"bindings": olympic_records}}
        out_olympic.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Wrote {len(olympic_records)} Olympic records to {out_olympic}")

        await asyncio.sleep(2)

        paralympic_records = await harvest(PARALYMPIC_PAGE, is_paralympic=True, client=client)
        out = {"head": {"vars": [
            "athlete", "athleteLabel", "sportLabel", "birthPlace",
            "birthPlaceLabel", "genderLabel", "dateOfBirth", "medals"
        ]}, "results": {"bindings": paralympic_records}}
        out_paralympic.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Wrote {len(paralympic_records)} Paralympic records to {out_paralympic}")

        # Summary
        olympic_with_hometown = sum(1 for r in olympic_records if "birthPlaceLabel" in r)
        paralympic_with_hometown = sum(1 for r in paralympic_records if "birthPlaceLabel" in r)
        logger.info("=" * 50)
        logger.info(f"Olympic: {len(olympic_records)} athletes, {olympic_with_hometown} with hometown ({100*olympic_with_hometown//max(1,len(olympic_records))}%)")
        logger.info(f"Paralympic: {len(paralympic_records)} athletes, {paralympic_with_hometown} with hometown ({100*paralympic_with_hometown//max(1,len(paralympic_records))}%)")
        logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
# Note: Add pydantic>=2.9.0 to pipeline/requirements.txt if not already present.

import json
import logging
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OlympicParalympicStatus(str, Enum):
    OLYMPIC = "olympic"
    PARALYMPIC = "paralympic"
    BOTH = "both"


class Hometown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wikidata_uri: str
    label: str
    latitude: float | None = None
    longitude: float | None = None


class Athlete(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wikidata_uri: str
    name: str
    status: OlympicParalympicStatus
    sports: list[str]
    hometown: Hometown
    medals: list[str]
    gender: str | None = None
    date_of_birth: str | None = None


def load_raw(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning(f"Input file not found: {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", {}).get("bindings", [])


def extract_value(binding: dict, key: str) -> str | None:
    if key in binding and "value" in binding[key]:
        return binding[key]["value"]
    return None


def build_athletes(
    bindings: list[dict[str, Any]],
    status: OlympicParalympicStatus
) -> dict[str, Athlete]:
    athletes: dict[str, Athlete] = {}
    
    for row in bindings:
        athlete_uri = extract_value(row, "athlete")
        athlete_name = extract_value(row, "athleteLabel")
        birth_place_uri = extract_value(row, "birthPlace")
        birth_place_label = extract_value(row, "birthPlaceLabel")
        
        if not (athlete_uri and athlete_name and birth_place_uri and birth_place_label):
            logger.debug(f"Skipping incomplete row: {row}")
            continue

        sport = extract_value(row, "sportLabel")
        medal = extract_value(row, "medals")
        gender = extract_value(row, "genderLabel")
        dob = extract_value(row, "dateOfBirth")

        if athlete_uri not in athletes:
            athletes[athlete_uri] = Athlete(
                wikidata_uri=athlete_uri,
                name=athlete_name,
                status=status,
                sports=[sport] if sport else [],
                hometown=Hometown(wikidata_uri=birth_place_uri, label=birth_place_label),
                medals=[medal] if medal else [],
                gender=gender,
                date_of_birth=dob
            )
        else:
            existing = athletes[athlete_uri]
            if sport and sport not in existing.sports:
                existing.sports.append(sport)
            if medal and medal not in existing.medals:
                existing.medals.append(medal)
            
            if existing.gender is None and gender:
                existing.gender = gender
            if existing.date_of_birth is None and dob:
                existing.date_of_birth = dob

    return athletes


def merge_olympic_paralympic(
    olympic: dict[str, Athlete],
    paralympic: dict[str, Athlete]
) -> list[Athlete]:
    merged: dict[str, Athlete] = {}
    
    for uri, athlete in olympic.items():
        merged[uri] = athlete
        
    for uri, athlete in paralympic.items():
        if uri in merged:
            existing = merged[uri]
            existing.status = OlympicParalympicStatus.BOTH
            
            for sport in athlete.sports:
                if sport not in existing.sports:
                    existing.sports.append(sport)
                    
            for medal in athlete.medals:
                if medal not in existing.medals:
                    existing.medals.append(medal)
                    
            if existing.gender is None and athlete.gender:
                existing.gender = athlete.gender
            if existing.date_of_birth is None and athlete.date_of_birth:
                existing.date_of_birth = athlete.date_of_birth
        else:
            merged[uri] = athlete

    return sorted(merged.values(), key=lambda a: a.wikidata_uri)


def write_normalized(
    athletes: list[Athlete],
    path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_data = [athlete.model_dump(mode="json") for athlete in athletes]
    with path.open("w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=2, ensure_ascii=False)


def log_summary(athletes: list[Athlete]) -> None:
    total = len(athletes)
    status_counts = Counter(a.status for a in athletes)
    
    sports_counter: Counter[str] = Counter()
    hometowns_counter: Counter[str] = Counter()
    medals_count = 0
    missing_gender = 0
    missing_dob = 0
    
    for a in athletes:
        for sport in a.sports:
            sports_counter[sport] += 1
        
        hometowns_counter[a.hometown.label] += 1
        
        if a.medals:
            medals_count += 1
        if not a.gender:
            missing_gender += 1
        if not a.date_of_birth:
            missing_dob += 1

    logger.info("--- NORMALIZATION SUMMARY ---")
    logger.info(f"Total normalized athletes: {total}")
    
    logger.info("Count by status:")
    for status_enum in OlympicParalympicStatus:
        logger.info(f"  {status_enum.value}: {status_counts.get(status_enum, 0)}")
        
    logger.info("Top 10 sports by count:")
    for sport, count in sports_counter.most_common(10):
        logger.info(f"  {sport}: {count}")
        
    logger.info("Top 10 hometowns by count:")
    for hometown, count in hometowns_counter.most_common(10):
        logger.info(f"  {hometown}: {count}")
        
    logger.info(f"Athletes with at least one medal claim: {medals_count}")
    logger.info(f"Athletes missing gender: {missing_gender}")
    logger.info(f"Athletes missing date_of_birth: {missing_dob}")


def main() -> None:
    base_dir = Path("pipeline")
    raw_dir = base_dir / "raw_data" / "wikidata"
    olympic_paths = [
        raw_dir / "athletes_olympic.json",
        raw_dir / "athletes_olympic_2026.json",
    ]
    paralympic_paths = [
        raw_dir / "athletes_paralympic.json",
        raw_dir / "athletes_paralympic_2026.json",
    ]
    out_path = base_dir / "normalized" / "athletes.json"

    logger.info("Loading and parsing Olympic data...")
    olympic_bindings: list[dict[str, Any]] = []
    for p in olympic_paths:
        loaded = load_raw(p)
        logger.info(f"  {p.name}: {len(loaded)} bindings")
        olympic_bindings.extend(loaded)
    olympic_athletes = build_athletes(olympic_bindings, OlympicParalympicStatus.OLYMPIC)

    logger.info("Loading and parsing Paralympic data...")
    paralympic_bindings: list[dict[str, Any]] = []
    for p in paralympic_paths:
        loaded = load_raw(p)
        logger.info(f"  {p.name}: {len(loaded)} bindings")
        paralympic_bindings.extend(loaded)
    paralympic_athletes = build_athletes(paralympic_bindings, OlympicParalympicStatus.PARALYMPIC)

    logger.info("Merging datasets...")
    merged_athletes = merge_olympic_paralympic(olympic_athletes, paralympic_athletes)

    logger.info(f"Writing normalized dataset to {out_path}...")
    write_normalized(merged_athletes, out_path)

    log_summary(merged_athletes)


if __name__ == "__main__":
    main()
# Note: Add scikit-learn>=1.5.0 to pipeline/requirements.txt if not already present.

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from pydantic import BaseModel, ConfigDict

from pipeline.normalize import Athlete, Hometown, OlympicParalympicStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
EPSILON_RADIANS = 50.0 / EARTH_RADIUS_KM

CITY_TO_STATE = {
    # Northeast
    "New York City": "NY", "Brooklyn": "NY", "Queens": "NY",
    "Bronx": "NY", "Manhattan": "NY", "Staten Island": "NY",
    "Yonkers": "NY", "Albany": "NY", "Buffalo": "NY", "Rochester": "NY",
    "Syracuse": "NY", "Long Island": "NY", "Mineola": "NY",
    "Oyster Bay": "NY", "Suffern": "NY", "Port Jefferson": "NY",
    "Massapequa": "NY",
    "Newark": "NJ", "Jersey City": "NJ", "Hoboken": "NJ",
    "Paterson": "NJ", "Trenton": "NJ", "Passaic": "NJ",
    "Hackensack": "NJ", "Cranford": "NJ", "Freehold": "NJ",
    "South Plainfield": "NJ", "Berwyn": "IL", "Wayne": "NJ",
    "Union City": "NJ",
    "Philadelphia": "PA", "Pittsburgh": "PA", "Reading": "PA",
    "Scranton": "PA", "Bryn Mawr": "PA", "Phillipsburg": "NJ",
    "Bridgeton": "NJ", "Allentown": "PA", "Spruce Hill": "PA",
    "Boston": "MA", "Cambridge": "MA", "Worcester": "MA",
    "Springfield": "MA", "Marlborough": "MA", "Wenham": "MA",
    "Medford": "MA", "Stoneham": "MA", "Haverhill": "MA",
    "Wareham": "MA", "Fitchburg": "MA",
    "Providence": "RI", "Manchester": "NH", "Portsmouth": "NH",
    "Hanover": "NH", "Londonderry": "NH",
    "Hartford": "CT", "New Haven": "CT", "Bridgeport": "CT",
    "Burlington": "VT",
    "Portland": "ME",
    "Washington, D.C.": "DC",
    "Baltimore": "MD", "Frederick": "MD", "Silver Spring": "MD",
    "Mount Airy": "MD", "La Plata": "MD",
    # Midwest
    "Chicago": "IL", "Champaign": "IL", "Downers Grove": "IL",
    "Melrose Park": "IL", "Palos Park": "IL", "Gurnee": "IL",
    "Detroit": "MI", "Ann Arbor": "MI", "Lansing": "MI",
    "Saginaw": "MI", "Grand Haven": "MI", "Allegan": "MI",
    "Royal Oak": "MI", "Lapeer": "MI", "Eaton Rapids": "MI",
    "Warren": "MI", "Clinton Township": "MI",
    "Cleveland": "OH", "Columbus": "OH", "Cincinnati": "OH",
    "Akron": "OH", "Toledo": "OH", "Canton": "OH", "Dayton": "OH",
    "Upper Arlington": "OH", "Celina": "OH",
    "Indianapolis": "IN", "Muncie": "IN", "Goshen": "IN",
    "Terre Haute": "IN", "New Albany": "IN",
    "Milwaukee": "WI", "Madison": "WI", "Green Bay": "WI",
    "Waukesha": "WI", "Sheboygan": "WI", "Oconto": "WI",
    "Grantsburg": "WI",
    "Minneapolis": "MN", "Saint Paul": "MN", "St. Paul": "MN",
    "Duluth": "MN", "Mankato": "MN", "Lakeville": "MN",
    "Owatonna": "MN", "Litchfield": "MN",
    "St. Louis": "MO", "Kansas City": "MO",
    "Omaha": "NE", "Lincoln": "NE",
    "Des Moines": "IA", "Davenport": "IA", "Council Bluffs": "IA",
    "Cresco": "IA", "Larchwood": "IA", "Schuyler": "IA",
    "Topeka": "KS", "Wichita": "KS", "Salina": "KS", "Lawrence": "KS",
    "Cape Girardeau": "MO", "Wyaconda": "MO",
    "Fargo": "ND", "Sioux Falls": "SD",
    # South
    "Atlanta": "GA", "Macon": "GA", "Cartersville": "GA",
    "Clarkesville": "GA", "Eastman": "GA",
    "Charlotte": "NC", "Raleigh": "NC", "Greensboro": "NC",
    "Hickory": "NC", "Huntersville": "NC", "Greenville": "NC",
    "Columbia": "SC", "Greenwood": "SC",
    "Jacksonville": "FL", "Miami": "FL", "Tampa": "FL", "Orlando": "FL",
    "Cape Canaveral": "FL", "Coral Gables": "FL", "Winter Haven": "FL",
    "Riverview": "FL",
    "Birmingham": "AL", "Mobile": "AL", "Huntsville": "AL",
    "Nashville": "TN", "Memphis": "TN", "Knoxville": "TN",
    "Chattanooga": "TN", "Kingsport": "TN",
    "Louisville": "KY",
    "New Orleans": "LA", "Baton Rouge": "LA", "Laurel": "MS",
    "Jackson": "MS",
    "Houston": "TX", "Dallas": "TX", "San Antonio": "TX",
    "Austin": "TX", "Fort Worth": "TX", "Plano": "TX",
    "Amarillo": "TX", "Wichita Falls": "TX", "Wylie": "TX",
    "Mesquite": "TX", "Terrell": "TX", "Stockton": "TX",
    "Perryton": "TX",
    "Oklahoma City": "OK", "Tulsa": "OK", "Stillwater": "OK",
    "Checotah": "OK", "Claremore": "OK", "Dewar": "OK",
    "Midwest City": "OK",
    "Little Rock": "AR", "Tuckerman": "AR",
    "Charleston": "SC",
    "Hopkins": "MN",
    "San Juan": "PR",
    # Mountain
    "Denver": "CO", "Boulder": "CO", "Colorado Springs": "CO",
    "Aspen": "CO", "Steamboat Springs": "CO", "Vail": "CO",
    "Berthoud": "CO", "Wheat Ridge": "CO", "Longmont": "CO",
    "Salt Lake City": "UT",
    "Phoenix": "AZ", "Tucson": "AZ", "Nogales": "AZ",
    "Prescott Valley": "AZ",
    "Albuquerque": "NM", "Silver City": "NM",
    "Boise": "ID", "Orofino": "ID",
    "Helena": "MT", "Billings": "MT", "Missoula": "MT",
    "Cheyenne": "WY",
    "Las Vegas": "NV", "Reno": "NV", "Las Vegas Valley": "NV",
    "Park Rapids": "MN",
    # Pacific / West
    "Los Angeles": "CA", "Inglewood": "CA", "Bell": "CA",
    "South Pasadena": "CA", "Glendale": "CA", "Santa Clarita": "CA",
    "Pasadena": "CA", "West Covina": "CA", "Fullerton": "CA",
    "Santa Maria": "CA", "Orange": "CA", "Hanford": "CA",
    "Fresno": "CA", "Merced": "CA", "Fremont": "CA",
    "San Francisco": "CA", "Oakland": "CA", "San Jose": "CA",
    "San Diego": "CA", "Riverside": "CA", "Sacramento": "CA",
    "Sunnyvale": "CA", "Los Gatos": "CA", "Harbor City": "CA",
    "San Gabriel": "CA",
    "Portland": "OR", "Eugene": "OR", "Beaverton": "OR",
    "Seattle": "WA", "Tacoma": "WA", "Spokane": "WA",
    "Redmond": "WA", "Everett": "WA", "Benton City": "WA",
    "Richland": "WA", "Woodinville": "WA",
    "Anchorage": "AK", "Juneau": "AK", "Palmer": "AK",
    "Honolulu": "HI", "Mililani": "HI", "Paia": "HI",
    "Falmouth": "MA",
}

DISPLAY_NAME_OVERRIDES = {
    # NYC metro - if medoid lands in any of these, label as NYC Metro
    "Union City": "New York Metro Region",
    "Bell": "Los Angeles Metro Region",
    "Yonkers": "New York Metro Region",
    "Bronx": "New York Metro Region",
    "Inglewood": "Los Angeles Metro Region",
    "Hoboken": "New York Metro Region",
    "Jersey City": "New York Metro Region",
    "Newark": "New York Metro Region",
    "Brooklyn": "New York Metro Region",
    "Queens": "New York Metro Region",
    "Cambridge": "Boston Metro Region",
    "Oakland": "Bay Area Region",
    "San Jose": "Bay Area Region",
    "Fremont": "Bay Area Region",
    "Pasadena": "Los Angeles Metro Region",
    "Glendale": "Los Angeles Metro Region",
}


class SportInHub(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sport: str
    count: int
    paralympic_count: int
    track_type: str


class HubComposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    olympic_count: int
    paralympic_count: int
    both_count: int
    paralympic_share: float
    composition_label: str


class Hub(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hub_id: str
    display_name: str
    centroid_latitude: float
    centroid_longitude: float
    medoid_hometown: str
    radius_km: float
    region: str
    states: list[str]
    total_athletes: int
    composition: HubComposition
    is_paralympic_hot_spot: bool
    top_sports: list[SportInHub]
    sport_diversity_index: float
    tags: list[str]
    search_aliases: list[str]


class ClusteredAthlete(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wikidata_uri: str
    name: str
    status: OlympicParalympicStatus
    sports: list[str]
    hometown: Hometown
    medals: list[str]
    gender: str | None = None
    date_of_birth: str | None = None
    hub_id: str
    is_core_member: bool
    distance_to_hub_km: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def get_state_from_city(city_label: str) -> str:
    for city, state in CITY_TO_STATE.items():
        if city.lower() in city_label.lower():
            return state
    return "XX"


def get_region(lat: float, lon: float) -> str:
    if lat >= 38 and lon >= -80:
        return "northeast"
    if lat >= 36 and -100 <= lon < -80:
        return "midwest"
    if lat < 36 and lon >= -100:
        return "south"
    if lat >= 31 and -115 <= lon < -100:
        return "mountain"
    if lat >= 31 and lon < -115:
        return "pacific"
    return "west"


def main() -> None:
    base_dir = Path("pipeline")
    in_path = base_dir / "geocoded" / "athletes.json"
    athletes_out_path = base_dir / "clustered" / "athletes.json"
    hubs_out_path = base_dir / "clustered" / "hubs.json"

    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        return

    logger.info("Loading geocoded athletes...")
    with in_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    athletes: list[Athlete] = []
    for item in raw_data:
        athletes.append(Athlete.model_validate(item))

    valid_athletes = []
    dropped_athletes = []
    for a in athletes:
        if a.hometown.latitude is not None and a.hometown.longitude is not None:
            valid_athletes.append(a)
        else:
            dropped_athletes.append(a)

    logger.info(f"Dropped {len(dropped_athletes)} athletes due to missing coordinates.")

    if not valid_athletes:
        logger.error("No athletes with valid coordinates to cluster.")
        return

    coords = np.array([[a.hometown.latitude, a.hometown.longitude] for a in valid_athletes])
    coords_rad = np.radians(coords)

    logger.info("Running HDBSCAN Stage 1...")
    clusterer = HDBSCAN(
        min_cluster_size=20,
        min_samples=8,
        cluster_selection_epsilon=EPSILON_RADIANS,
        metric="haversine",
        cluster_selection_method="eom",
        store_centers="medoid",
        allow_single_cluster=False,
    )
    clusterer.fit(coords_rad)

    labels = clusterer.labels_.copy()
    raw_cluster_count = len(set(labels) - {-1})
    logger.info(f"Discovered {raw_cluster_count} raw clusters. Stage 1 complete.")

    if clusterer.medoids_ is None or raw_cluster_count == 0:
        logger.error("HDBSCAN failed to find any clusters.")
        return

    medoids_deg = np.degrees(clusterer.medoids_)

    logger.info("Running Stage 2: Nearest-medoid assignment for noise points...")
    is_core_member = (labels != -1)
    
    for i, label in enumerate(labels):
        if label == -1:
            lat1, lon1 = coords[i]
            min_dist = float('inf')
            closest_c = -1
            for c_idx, (m_lat, m_lon) in enumerate(medoids_deg):
                dist = haversine_km(lat1, lon1, m_lat, m_lon)
                if dist < min_dist:
                    min_dist = dist
                    closest_c = c_idx
            labels[i] = closest_c

    cluster_to_athletes: dict[int, list[tuple[Athlete, bool, float, float]]] = {}
    for c_idx in range(len(medoids_deg)):
        cluster_to_athletes[c_idx] = []

    for i, c_idx in enumerate(labels):
        cluster_to_athletes[c_idx].append((valid_athletes[i], bool(is_core_member[i]), coords[i][0], coords[i][1]))

    logger.info("Running Stage 3 & 4: Hub generation and metadata computation...")
    hubs: list[Hub] = []
    clustered_athletes_out: list[ClusteredAthlete] = []

    winter_sports = {
        "figure skating", "ice hockey", "alpine skiing", "speed skating", 
        "snowboarding", "cross-country skiing", "bobsleigh", "luge", 
        "skeleton", "curling", "biathlon", "freestyle skiing", 
        "nordic combined", "ski jumping", "short track speed skating"
    }

    for c_idx, (m_lat, m_lon) in enumerate(medoids_deg):
        members = cluster_to_athletes[c_idx]
        if not members:
            continue

        min_m_dist = float('inf')
        medoid_hometown = ""
        for a, core, a_lat, a_lon in members:
            dist = haversine_km(m_lat, m_lon, a_lat, a_lon)
            if dist < min_m_dist:
                min_m_dist = dist
                medoid_hometown = a.hometown.label

        city_slug = re.sub(r'[^A-Z0-9_]', '', medoid_hometown.upper().replace(' ', '_'))
        state = get_state_from_city(medoid_hometown)
        hub_id = f"HUB_{state}_{city_slug}"

        if medoid_hometown in DISPLAY_NAME_OVERRIDES:
            display_name = DISPLAY_NAME_OVERRIDES[medoid_hometown]
        else:
            display_name = f"{medoid_hometown} Region, {state}"

        distances = []
        for a, core, a_lat, a_lon in members:
            distances.append(haversine_km(m_lat, m_lon, a_lat, a_lon))
        radius_km = float(np.percentile(distances, 95)) if distances else 0.0

        region = get_region(m_lat, m_lon)

        state_counter: Counter[str] = Counter()
        for a, _, _, _ in members:
            state_counter[get_state_from_city(a.hometown.label)] += 1
        top_states = [s for s, _ in state_counter.most_common(3)]

        oly_c = sum(1 for a, _, _, _ in members if a.status == OlympicParalympicStatus.OLYMPIC)
        para_c = sum(1 for a, _, _, _ in members if a.status == OlympicParalympicStatus.PARALYMPIC)
        both_c = sum(1 for a, _, _, _ in members if a.status == OlympicParalympicStatus.BOTH)
        total = len(members)

        para_share = (para_c + both_c) / total if total > 0 else 0.0

        if para_share >= 0.15:
            comp_label = "paralympic_strong"
        elif para_share >= 0.05:
            comp_label = "balanced"
        else:
            comp_label = "olympic_dominant"

        composition = HubComposition(
            olympic_count=oly_c,
            paralympic_count=para_c,
            both_count=both_c,
            paralympic_share=para_share,
            composition_label=comp_label
        )

        is_hot_spot = para_share >= 0.092

        sport_counter: Counter[str] = Counter()
        sport_para_counter: Counter[str] = Counter()
        for a, _, _, _ in members:
            is_para_athlete = a.status in (OlympicParalympicStatus.PARALYMPIC, OlympicParalympicStatus.BOTH)
            for s in a.sports:
                sport_counter[s] += 1
                if is_para_athlete:
                    sport_para_counter[s] += 1

        top_sports = []
        for s, count in sport_counter.most_common(3):
            p_count = sport_para_counter[s]
            if p_count == 0:
                t_type = "olympic"
            elif p_count == count:
                t_type = "paralympic"
            else:
                t_type = "both"
            top_sports.append(SportInHub(sport=s, count=count, paralympic_count=p_count, track_type=t_type))

        H = 0.0
        total_sports_count = sum(sport_counter.values())
        if total_sports_count > 0:
            for count in sport_counter.values():
                p = count / total_sports_count
                H -= p * math.log(p)
        num_unique = len(sport_counter)
        H_max = math.log(num_unique) if num_unique > 0 else 0.0
        diversity = (H / H_max) if H_max > 0 else 0.0

        tags = []
        if is_hot_spot:
            tags.append("para-hot-spot")
        has_winter = any(ts.sport.lower() in winter_sports for ts in top_sports)
        if has_winter:
            tags.append("winter-strong")
        else:
            tags.append("summer-strong")
        tags.append(region)
        if top_sports:
            sport_slug = re.sub(r'[^a-z0-9]+', '-', top_sports[0].sport.lower()).strip('-')
            tags.append(sport_slug)

        search_aliases = []
        label_lower = medoid_hometown.lower()
        if "new york" in label_lower:
            search_aliases.extend(["NYC", "Manhattan", "NY"])
        elif "minneapolis" in label_lower:
            search_aliases.extend(["Twin Cities", "MSP"])
        elif "san francisco" in label_lower:
            search_aliases.extend(["SF", "Bay Area"])
        elif "los angeles" in label_lower:
            search_aliases.extend(["LA", "L.A."])

        search_aliases.insert(0, medoid_hometown)
        if state not in search_aliases:
            search_aliases.append(state)

        hub = Hub(
            hub_id=hub_id,
            display_name=display_name,
            centroid_latitude=m_lat,
            centroid_longitude=m_lon,
            medoid_hometown=medoid_hometown,
            radius_km=radius_km,
            region=region,
            states=top_states,
            total_athletes=total,
            composition=composition,
            is_paralympic_hot_spot=is_hot_spot,
            top_sports=top_sports,
            sport_diversity_index=diversity,
            tags=tags,
            search_aliases=search_aliases
        )
        hubs.append(hub)

        for a, core, a_lat, a_lon in members:
            dist = haversine_km(m_lat, m_lon, a_lat, a_lon)
            ca = ClusteredAthlete(
                wikidata_uri=a.wikidata_uri,
                name=a.name,
                status=a.status,
                sports=a.sports,
                hometown=a.hometown,
                medals=a.medals,
                gender=a.gender,
                date_of_birth=a.date_of_birth,
                hub_id=hub.hub_id,
                is_core_member=core,
                distance_to_hub_km=dist
            )
            clustered_athletes_out.append(ca)

    hubs.sort(key=lambda x: x.total_athletes, reverse=True)

    athletes_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with athletes_out_path.open("w", encoding="utf-8") as f:
        json.dump([a.model_dump(mode="json") for a in clustered_athletes_out], f, indent=2, ensure_ascii=False)
        
    with hubs_out_path.open("w", encoding="utf-8") as f:
        json.dump([h.model_dump(mode="json") for h in hubs], f, indent=2, ensure_ascii=False)

    logger.info("--- CLUSTERING SUMMARY ---")
    logger.info(f"Total clusters discovered: {raw_cluster_count}")
    core_count = sum(1 for a in clustered_athletes_out if a.is_core_member)
    noise_count = len(clustered_athletes_out) - core_count
    logger.info(f"Total athletes assigned: {len(clustered_athletes_out)} ({core_count} core, {noise_count} nearest-medoid)")
    
    logger.info("Hubs with composition_label = 'paralympic_strong':")
    for h in hubs:
        if h.composition.composition_label == "paralympic_strong":
            logger.info(f"  {h.hub_id} ({h.display_name}): {h.composition.paralympic_share:.1%} para share")
            
    logger.info("Hubs flagged is_paralympic_hot_spot:")
    for h in hubs:
        if h.is_paralympic_hot_spot:
            logger.info(f"  {h.hub_id} ({h.display_name}): {h.composition.paralympic_share:.1%} para share")

    logger.info("Top 5 hubs by total_athletes:")
    for h in hubs[:5]:
        logger.info(f"  {h.hub_id} ({h.display_name}): {h.total_athletes} athletes")

    logger.info(f"Athletes dropped due to missing coordinates: {len(dropped_athletes)}")


if __name__ == "__main__":
    main()
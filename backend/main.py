import asyncio
import json
import base64
import logging
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from google import genai
from google.genai import types as genai_types

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# v2 build

VOICE_MODEL_ID = os.getenv("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-native-audio")
VOICE_TTS_MODEL_ID = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
VOICE_NAME = os.getenv("GEMINI_VOICE_NAME", "Kore")
VOICE_LOCATION = os.getenv("GEMINI_LIVE_LOCATION", "us-central1")
VOICE_TTS_MAX_CHARS = int(os.getenv("GEMINI_VOICE_TTS_MAX_CHARS", "900"))

# Approximate bounding boxes for all 50 US states + DC + territories.
# Format: (lat_min, lat_max, lon_min, lon_max, state_code)
# Order matters: smaller/more specific bboxes FIRST so they match
# before larger ones overlap them. Tested for non-overlap across
# major metros ,  borderline points (within 0.2 degrees of multiple
# state lines) get the FIRST matching bbox.
STATE_BBOXES = [
    # Territories first (most specific lat/lon)
    (-14.55, -14.10, -171.10, -169.40, "AS"), # American Samoa
    (13.20, 13.70, 144.60, 145.00, "GU"),    # Guam
    (14.10, 20.55, 144.85, 146.10, "MP"),    # Northern Mariana Islands
    (17.65, 18.45, -65.10, -64.55, "VI"),    # US Virgin Islands
    (17.85, 18.55, -67.30, -65.20, "PR"),    # Puerto Rico
    # Hawaii ,  distinct island chain
    (18.90, 22.30, -160.30, -154.80, "HI"),
    # Alaska ,  wide range
    (51.00, 71.50, -180.00, -130.00, "AK"),
    # DC ,  small, must come BEFORE MD and VA
    (38.79, 39.00, -77.12, -76.91, "DC"),
    # New England ,  small states FIRST so they don't get swallowed by NY/NH
    (41.30, 42.05, -71.90, -71.10, "RI"), # Rhode Island
    (41.00, 42.05, -73.75, -71.79, "CT"), # Connecticut
    (42.00, 42.90, -73.50, -69.90, "MA"), # Massachusetts
    (42.70, 45.00, -72.55, -70.55, "NH"), # New Hampshire
    (42.73, 45.02, -73.43, -71.46, "VT"), # Vermont
    (43.05, 47.45, -71.10, -66.95, "ME"), # Maine
    # Mid-Atlantic
    (38.45, 39.72, -75.79, -74.98, "DE"),
    (37.90, 39.73, -79.49, -75.04, "MD"),
    (38.90, 41.36, -75.56, -73.90, "NJ"),
    (39.72, 42.27, -80.52, -74.69, "PA"),
    (40.50, 45.02, -79.76, -71.85, "NY"),
    (36.54, 39.47, -83.68, -75.24, "VA"),
    (37.20, 40.64, -82.64, -77.72, "WV"),
    # South
    (33.78, 36.59, -84.32, -75.46, "NC"),
    (32.03, 35.22, -83.35, -78.54, "SC"),
    (30.36, 35.00, -85.61, -80.84, "GA"),
    (24.40, 31.00, -87.63, -79.97, "FL"),
    (30.20, 35.01, -88.47, -84.89, "AL"),
    (35.00, 36.68, -90.31, -81.65, "TN"),
    (36.50, 39.15, -89.57, -81.96, "KY"),
    (30.17, 35.00, -91.65, -88.10, "MS"),
    (28.93, 33.02, -94.05, -88.76, "LA"),
    (33.00, 36.50, -94.62, -89.64, "AR"),
    # Midwest
    (33.62, 37.00, -103.00, -94.43, "OK"), # OK lat 33.6-37 (real bounds)
    (25.84, 36.50, -106.65, -93.51, "TX"),
    (36.99, 40.00, -102.05, -94.59, "KS"),
    (40.00, 43.00, -104.05, -95.30, "NE"),
    (37.00, 40.62, -95.77, -89.10, "MO"),
    (40.38, 43.50, -96.64, -90.14, "IA"),
    (42.49, 49.40, -97.24, -89.50, "MN"),
    (42.49, 47.08, -92.89, -86.25, "WI"),
    (41.70, 48.31, -90.42, -82.41, "MI"),
    (37.77, 42.51, -91.51, -87.50, "IL"),
    (37.77, 41.76, -88.10, -84.78, "IN"),
    (38.40, 42.00, -84.82, -80.52, "OH"),
    (45.94, 49.00, -104.05, -96.55, "ND"),
    (42.48, 45.94, -104.06, -96.44, "SD"),
    # Mountain West
    (44.40, 49.00, -116.05, -104.04, "MT"),
    (44.00, 49.00, -117.24, -111.05, "ID"),
    (40.99, 45.01, -111.06, -104.05, "WY"),
    (36.99, 41.00, -109.06, -102.04, "CO"),
    (36.99, 42.00, -114.06, -109.04, "UT"),
    (35.00, 42.00, -120.01, -114.04, "NV"),
    (31.33, 37.00, -114.82, -109.04, "AZ"),
    (31.33, 37.00, -109.05, -103.00, "NM"),
    # Pacific
    (45.54, 49.00, -124.84, -116.91, "WA"),
    (41.99, 46.30, -124.57, -116.46, "OR"),
    (32.53, 42.00, -124.49, -114.13, "CA"),
]

STATE_CODE_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida",
    "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "PR": "Puerto Rico", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
    "WY": "Wyoming",
    "AS": "American Samoa", "GU": "Guam",
    "MP": "Northern Mariana Islands", "VI": "U.S. Virgin Islands",
}
STATE_NAME_TO_CODE = {
    _name.lower(): _code for _code, _name in STATE_CODE_TO_NAME.items()
}

STATE_GEOJSON_PATHS = [
    Path(__file__).resolve().parents[1] / "pipeline" / "geo" / "us-states.json",
    Path(__file__).resolve().parent / "pipeline" / "geo" / "us-states.json",
    Path("/app") / "pipeline" / "geo" / "us-states.json",
]
_STATE_POLYGONS: list[tuple[str, dict[str, Any]]] | None = None


def _load_state_polygons() -> list[tuple[str, dict[str, Any]]]:
    """Load simplified state polygons used by the frontend map.

    The previous state assignment used rectangular bounding boxes. That made
    plotted constellation dots disagree with state modal counts around irregular
    borders such as Idaho, Nevada, and Oregon. This classifier uses the same
    GeoJSON family the frontend renders, then falls back to bounding boxes for
    territories and simplified-island edge cases.
    """
    global _STATE_POLYGONS
    if _STATE_POLYGONS is not None:
        return _STATE_POLYGONS
    geojson_path = next((path for path in STATE_GEOJSON_PATHS if path.exists()), None)
    if geojson_path is None:
        logger.warning(
            "State GeoJSON not found in candidate paths "
            f"{[str(path) for path in STATE_GEOJSON_PATHS]}; using bbox fallback only."
        )
        _STATE_POLYGONS = []
        return _STATE_POLYGONS

    with geojson_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    polygons: list[tuple[str, dict[str, Any]]] = []
    for feature in geojson.get("features", []):
        name = str(feature.get("properties", {}).get("name") or "").lower()
        code = STATE_NAME_TO_CODE.get(name)
        geometry = feature.get("geometry")
        if code and geometry:
            polygons.append((code, geometry))
    _STATE_POLYGONS = polygons
    return polygons


def _point_on_segment(
    lon: float,
    lat: float,
    a: list[float],
    b: list[float],
    eps: float = 1e-10,
) -> bool:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    squared_len = (bx - ax) ** 2 + (by - ay) ** 2
    if squared_len <= eps:
        return abs(lon - ax) <= eps and abs(lat - ay) <= eps
    cross = (lon - ax) * (by - ay) - (lat - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (lon - ax) * (bx - ax) + (lat - ay) * (by - ay)
    if dot < -eps:
        return False
    return dot <= squared_len + eps


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    j = len(ring) - 1
    for i in range(len(ring)):
        pi = ring[i]
        pj = ring[j]
        if _point_on_segment(lon, lat, pi, pj):
            return True
        xi, yi = float(pi[0]), float(pi[1])
        xj, yj = float(pj[0]), float(pj[1])
        intersects = (yi > lat) != (yj > lat)
        if intersects:
            x_at_lat = ((xj - xi) * (lat - yi) / ((yj - yi) or 1e-30)) + xi
            if lon < x_at_lat:
                inside = not inside
        j = i
    return inside


def _point_in_polygon(lon: float, lat: float, polygon: list[list[list[float]]]) -> bool:
    if not polygon or not _point_in_ring(lon, lat, polygon[0]):
        return False
    return not any(_point_in_ring(lon, lat, hole) for hole in polygon[1:])


def _state_from_geojson(lat: float, lon: float) -> str | None:
    for code, geometry in _load_state_polygons():
        geom_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        polygons = [coordinates] if geom_type == "Polygon" else coordinates
        for polygon in polygons:
            if _point_in_polygon(lon, lat, polygon):
                return code
    return None


def state_from_latlon(lat: float, lon: float) -> str:
    """Returns 2-letter US state/territory code for a given lat/lon.
    Returns 'XX' as a last-resort fallback only when coordinates are
    outside all known US bounding boxes."""
    polygon_state = _state_from_geojson(lat, lon)
    if polygon_state:
        return polygon_state
    for lat_min, lat_max, lon_min, lon_max, code in STATE_BBOXES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return code
    return "XX"


class HubComposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    olympic_count: int
    paralympic_count: int
    both_count: int
    paralympic_share: float
    composition_label: str


class SportInHub(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sport: str
    count: int
    paralympic_count: int
    track_type: str


class Hub(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hub_id: str
    display_name: str
    centroid_latitude: float
    centroid_longitude: float
    medoid_hometown: str
    radius_km: float
    region: str
    region_name: str
    macro_region: str
    states: list[str]
    total_athletes: int
    composition: HubComposition
    is_paralympic_hot_spot: bool
    top_sports: list[SportInHub]
    sport_diversity_index: float
    tags: list[str]
    search_aliases: list[str]


class ClimateData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annual_avg_temp_f: Optional[float] = None
    annual_precipitation_in: Optional[float] = None
    annual_sunshine_hours: Optional[float] = None
    elevation_ft: Optional[float] = None


class HubNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hub_id: str
    display_name: str
    headline: str
    summary: str
    paralympic_callout: Optional[str] = None
    top_sport_phrase: str
    confidence_qualifier: str
    geographic_context: Optional[str] = None
    climate: Optional[ClimateData] = None


class AthleteGeoPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hub_id: str
    lat: float
    lon: float
    status: Literal["olympic", "paralympic", "both"]
    state: str


class StateAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str
    total_athletes: int
    olympic_count: int
    paralympic_count: int
    both_count: int
    paralympic_share: float


class SelectHubAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["SELECT_HUB"] = "SELECT_HUB"
    hub_id: str


class FilterUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    macro_region: Optional[str] = None
    region_name: Optional[str] = None
    paralympic_focus: Optional[bool] = None
    sport_category: Optional[Literal["summer", "winter", "all"]] = None


class SetFilterAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["SET_FILTER"] = "SET_FILTER"
    filter: FilterUpdate


class FilterMapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    macro_region: Optional[Literal[
        "Northeast", "Mid-Atlantic", "South", "Midwest",
        "Southwest", "Mountain West", "Pacific", "Alaska",
        "Hawaii", "Territories"
    ]] = None
    region_name: Optional[str] = None
    paralympic_focus: Optional[bool] = None
    sport_category: Optional[Literal["summer", "winter", "all"]] = None


class ZoomToHubRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hub_id: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)
    session_id: Optional[str] = None


class ChatToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal[
        "select_hub",
        "filter_to_paralympic",
        "zoom_to_hub",
        "reset_view",
        "select_state",
        "query_data",
        "explain_map",
        "focus_hometown",
    ]
    args: dict


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)


QUERY_TYPES = [
    "summary",
    "rank_list",
    "entity_rank",
    "state_profile",
    "hub_profile",
    "compare_states",
    "compare_hubs",
    "all_hot_spots",
    "hubs_by_sport",
    "hubs_by_macro_region",
]
METRICS = [
    "total_athletes",
    "olympic_athletes",
    "paralympic_athletes",
    "paralympic_share",
    "sport_count",
]
ENTITY_TYPES = ["hub", "state"]
MACRO_REGIONS = [
    "Northeast", "Mid-Atlantic", "South", "Midwest",
    "Southwest", "Mountain West", "Pacific", "Alaska",
    "Hawaii", "Territories",
]


def _build_chatbot_tools() -> genai_types.Tool:
    hub_ids = sorted(_state["hubs_by_id"].keys())
    state_codes = sorted({s.state for s in _state["state_aggregates"]})

    return genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name="select_hub",
                description="Select a specific hometown hub and open its profile. Use when the user names a hub, city, or regional label and wants to inspect it on the map.",
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties={
                        "hub_id": genai_types.Schema(
                            type="STRING",
                            enum=hub_ids,
                            description="Exact hub_id from the current map dataset.",
                        ),
                    },
                    required=["hub_id"],
                ),
            ),
            genai_types.FunctionDeclaration(
                name="zoom_to_hub",
                description="Zoom the map to a specific hometown hub. Use when the user asks to go to, zoom to, or focus on a hub.",
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties={
                        "hub_id": genai_types.Schema(
                            type="STRING",
                            enum=hub_ids,
                            description="Exact hub_id from the current map dataset.",
                        ),
                    },
                    required=["hub_id"],
                ),
            ),
            genai_types.FunctionDeclaration(
                name="filter_to_paralympic",
                description="Filter the map to Paralympic Hot Spots. Optionally narrow to one macro region.",
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties={
                        "macro_region": genai_types.Schema(
                            type="STRING",
                            enum=MACRO_REGIONS,
                            description="Optional macro region filter. Omit to show all Hot Spots.",
                        ),
                    },
                ),
            ),
            genai_types.FunctionDeclaration(
                name="select_state",
                description="Open the state info panel for a US state, territory, or DC. Use when the user asks about a specific state and is not asking for a ranking or comparison.",
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties={
                        "state_code": genai_types.Schema(
                            type="STRING",
                            enum=state_codes,
                            description="2-letter US state or territory code from the current dataset.",
                        ),
                    },
                    required=["state_code"],
                ),
            ),
            genai_types.FunctionDeclaration(
                name="reset_view",
                description="Clear the current hub/state selection and filters, then return the map to the default national view.",
                parameters=genai_types.Schema(type="OBJECT", properties={}),
            ),
            genai_types.FunctionDeclaration(
                name="explain_map",
                description="Explain how to read the map legend, including athlete dots, hub circles, Hot Spots, state shading, colors, and Alaska/Hawaii/Puerto Rico insets.",
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties={
                        "topic": genai_types.Schema(
                            type="STRING",
                            description="Optional focus such as dots, circles, colors, red, blue, territories, Hot Spots, or legend.",
                        ),
                    },
                ),
            ),
            genai_types.FunctionDeclaration(
                name="focus_hometown",
                description="Resolve an aggregate hometown lookup and move the map to that hometown when possible. Use for questions like 'how many athletes are from Boise, Idaho?' or 'tell me about my hometown Park City'. Never returns individual athlete names.",
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties={
                        "hometown": genai_types.Schema(
                            type="STRING",
                            description="City or hometown name from the user's question.",
                        ),
                        "state_code": genai_types.Schema(
                            type="STRING",
                            enum=state_codes,
                            description="Optional state or territory code when the user provides one.",
                        ),
                    },
                    required=["hometown"],
                ),
            ),
            genai_types.FunctionDeclaration(
                name="query_data",
                description="Ask the Hometown Success Engine data layer for rankings, profiles, comparisons, totals, sports, regions, and Hot Spot intelligence without moving the map.",
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties={
                        "query_type": genai_types.Schema(
                            type="STRING",
                            enum=QUERY_TYPES,
                            description="summary, rank_list, entity_rank, state_profile, hub_profile, compare_states, compare_hubs, all_hot_spots, hubs_by_sport, or hubs_by_macro_region.",
                        ),
                        "entity_type": genai_types.Schema(
                            type="STRING",
                            enum=ENTITY_TYPES,
                            description="hub or state. Required for rank_list and entity_rank.",
                        ),
                        "metric": genai_types.Schema(
                            type="STRING",
                            enum=METRICS,
                            description="Ranking or comparison metric.",
                        ),
                        "state_code": genai_types.Schema(
                            type="STRING",
                            enum=state_codes,
                            description="Single state or territory code for state_profile or entity_rank.",
                        ),
                        "state_codes": genai_types.Schema(
                            type="ARRAY",
                            items=genai_types.Schema(type="STRING", enum=state_codes),
                            description="State or territory codes for compare_states.",
                        ),
                        "hub_id": genai_types.Schema(
                            type="STRING",
                            enum=hub_ids,
                            description="Single hub_id for hub_profile or entity_rank.",
                        ),
                        "hub_ids": genai_types.Schema(
                            type="ARRAY",
                            items=genai_types.Schema(type="STRING", enum=hub_ids),
                            description="Hub IDs for compare_hubs.",
                        ),
                        "sport": genai_types.Schema(
                            type="STRING",
                            description="Sport name for sport_count rankings or hubs_by_sport.",
                        ),
                        "macro_region": genai_types.Schema(
                            type="STRING",
                            enum=MACRO_REGIONS,
                            description="Macro region for hubs_by_macro_region.",
                        ),
                        "limit": genai_types.Schema(
                            type="INTEGER",
                            description="How many rows to return. Default 5, max 20.",
                        ),
                        "min_athletes": genai_types.Schema(
                            type="INTEGER",
                            description="Minimum athletes for share rankings. State Paralympic-share rankings default to 25.",
                        ),
                    },
                    required=["query_type"],
                ),
            ),
        ]
    )


CHATBOT_SYSTEM_PROMPT_TEMPLATE = """You are Gemini, the data guide for the Hometown Success Engine, an interactive map of where Team USA athletes are from. The current map shows {athlete_count:,} mapped Olympians and Paralympians across {hub_count} hometown hubs from Tokyo 2020 through Milan-Cortina 2026.

Current national Paralympic baseline: {baseline_pct:.1f}%.
Paralympic Hot Spot threshold: {hot_spot_threshold_pct:.1f}% Paralympic share or higher.
Current Paralympic Hot Spots: {hot_spot_count}.

# YOUR JOB
Help users explore the map and understand the data. Call tools to move the map or answer analyst questions, then explain the result with specific numbers from the tool output.

# TOOL RULES
- If the user asks to show, highlight, filter, or view Paralympic Hot Spots, call filter_to_paralympic.
- If the user asks to reset, clear, start over, or go back to the national view, call reset_view.
- If the user asks what the dots, circles, colors, legend, state shading, territory insets, Alaska/Hawaii/Puerto Rico insets, or Hot Spots mean, call explain_map.
- If the user asks how many athletes are from a hometown, asks about "my hometown", or gives a city that is not one of the 40 hub names, call focus_hometown with the city and state code if provided.
- If the user names a state and is not asking for rank or comparison, call select_state.
- If the user names a city, hub, or regional label and is not asking for rank or comparison, call select_hub or zoom_to_hub.
- If the user asks for rankings, totals, comparisons, profiles, sports, macro regions, or aggregate questions, call query_data.
- For "what rank is X" questions, call query_data with query_type entity_rank, entity_type hub or state, and the requested metric.
- For "rank/list/top" questions, call query_data with query_type rank_list, entity_type hub or state, metric, and limit.
- Always cite exact numbers from the tool result.
- Keep replies concise: usually 2-3 sentences unless the user asks for a list.
- Use conditional phrasing such as "could help find", "may foster", "may explain", or "is associated with". Never say a place produces athletes.
- Do not name individual athletes.
- Focus on mapped hometown hub athlete counts, not medal counts.

# CURRENT PARALYMPIC HOT SPOTS
{hot_spot_lines}

# CURRENT HUB LOOKUP
{hub_lookup_lines}
"""

_state: dict[str, Any] = {
    "hubs": [],
    "hubs_by_id": {},
    "narratives": {},
    "athletes_geo_points": [],
    "state_aggregates": [],
    "hometowns": [],
    "hometowns_by_key": {},
    "hometowns_by_name": defaultdict(list),
}

PARALYMPIC_HOT_SPOT_THRESHOLD_PCT = 7.5
CHAT_SESSION_TTL_SECONDS = 30 * 60
CHAT_SESSION_MAX_TURNS = 12
_chat_sessions: dict[str, dict[str, Any]] = {}


def _dataset_stats() -> dict[str, Any]:
    hubs = _state["hubs"]
    total_athletes = sum(h.total_athletes for h in hubs)
    total_para = sum(
        h.composition.paralympic_count + h.composition.both_count
        for h in hubs
    )
    hot_spots = [h for h in hubs if h.is_paralympic_hot_spot]
    baseline_pct = (total_para / total_athletes * 100) if total_athletes else 0.0
    return {
        "athlete_count": total_athletes,
        "hub_count": len(hubs),
        "hot_spot_count": len(hot_spots),
        "baseline_pct": baseline_pct,
    }


def _build_chatbot_system_prompt() -> str:
    stats = _dataset_stats()
    hot_spots = sorted(
        [h for h in _state["hubs"] if h.is_paralympic_hot_spot],
        key=lambda h: -h.composition.paralympic_share,
    )
    hot_spot_lines = "\n".join(
        f"- {h.hub_id}: {h.display_name}, "
        f"{h.composition.paralympic_share * 100:.1f}% Paralympic, "
        f"{h.total_athletes} athletes"
        for h in hot_spots
    ) or "- None"
    hub_lookup_lines = "\n".join(
        f"- {h.hub_id}: {h.display_name}; medoid {h.medoid_hometown}; "
        f"aliases {', '.join(h.search_aliases)}"
        for h in _state["hubs"]
    )
    return CHATBOT_SYSTEM_PROMPT_TEMPLATE.format(
        athlete_count=stats["athlete_count"],
        hub_count=stats["hub_count"],
        hot_spot_count=stats["hot_spot_count"],
        hot_spot_threshold_pct=PARALYMPIC_HOT_SPOT_THRESHOLD_PCT,
        baseline_pct=stats["baseline_pct"],
        hot_spot_lines=hot_spot_lines,
        hub_lookup_lines=hub_lookup_lines,
    )


def _limit(value: Any, default: int = 5) -> int:
    try:
        return max(1, min(int(value or default), 20))
    except (TypeError, ValueError):
        return default


def _metric_label(metric: str) -> str:
    return {
        "total_athletes": "total athletes",
        "olympic_athletes": "Olympians",
        "paralympic_athletes": "Paralympians",
        "paralympic_share": "Paralympic share",
        "sport_count": "sport count",
    }.get(metric, metric.replace("_", " "))


def _normalize_metric(metric: str | None, fallback: str = "total_athletes") -> str:
    value = (metric or "").lower().strip()
    aliases = {
        "total": "total_athletes",
        "overall": "total_athletes",
        "overall_athletes": "total_athletes",
        "athletes": "total_athletes",
        "olympic": "olympic_athletes",
        "olympians": "olympic_athletes",
        "paralympic": "paralympic_athletes",
        "paralympians": "paralympic_athletes",
        "para": "paralympic_athletes",
        "share": "paralympic_share",
        "paralympic_percentage": "paralympic_share",
        "percentage": "paralympic_share",
        "sport": "sport_count",
        "sports": "sport_count",
    }
    return aliases.get(value, value if value in METRICS else fallback)


def _metric_from_text(message: str) -> str:
    msg = message.lower()
    if "paralympic share" in msg or "para share" in msg or "percentage" in msg:
        return "paralympic_share"
    if "paralympic" in msg or "paralympian" in msg or " para " in f" {msg} ":
        return "paralympic_athletes"
    if "olympic" in msg or "olympian" in msg:
        return "olympic_athletes"
    if "sport" in msg or "ski" in msg or "swim" in msg:
        return "sport_count"
    return "total_athletes"


def _hub_para_count(hub: Hub) -> int:
    return hub.composition.paralympic_count + hub.composition.both_count


def _state_para_count(state: StateAggregate) -> int:
    return state.paralympic_count + state.both_count


def _sport_query(value: str | None) -> str:
    sport = (value or "").lower().strip().rstrip("?.!")
    aliases = {
        "skiing": "ski",
        "ski": "ski",
        "swimming": "swim",
        "track": "athletics",
        "track and field": "athletics",
    }
    return aliases.get(sport, sport)


def _display_sport(value: str | None) -> str:
    sport = (value or "").strip()
    labels = {
        "Winter Sport": "winter sports",
        "association football": "soccer",
        "cycle sport": "cycling",
        "competitive swimming": "swimming",
        "amateur wrestling": "wrestling",
        "shooting sports": "shooting",
    }
    return labels.get(sport, sport)


def _hub_sport_count(hub: Hub, sport: str | None) -> int:
    query = _sport_query(sport)
    if not query:
        return 0
    total = 0
    for sp in hub.top_sports:
        if query in sp.sport.lower():
            total += sp.count
    return total


def _hub_metric_value(hub: Hub, metric: str, sport: str | None = None) -> float:
    if metric == "total_athletes":
        return float(hub.total_athletes)
    if metric == "olympic_athletes":
        return float(hub.composition.olympic_count)
    if metric == "paralympic_athletes":
        return float(_hub_para_count(hub))
    if metric == "paralympic_share":
        return float(hub.composition.paralympic_share)
    if metric == "sport_count":
        return float(_hub_sport_count(hub, sport))
    return float(hub.total_athletes)


def _state_metric_value(state: StateAggregate, metric: str) -> float:
    if metric == "total_athletes":
        return float(state.total_athletes)
    if metric == "olympic_athletes":
        return float(state.olympic_count)
    if metric == "paralympic_athletes":
        return float(_state_para_count(state))
    if metric == "paralympic_share":
        return float(state.paralympic_share)
    return float(state.total_athletes)


def _format_metric_value(value: float, metric: str) -> str:
    if metric == "paralympic_share":
        return f"{value * 100:.1f}%"
    return f"{int(value):,}"


def _state_name(code: str) -> str:
    return STATE_CODE_TO_NAME.get(code, code)


def _ranked_hubs(
    metric: str,
    sport: str | None = None,
    min_athletes: int | None = None,
) -> list[Hub]:
    metric = _normalize_metric(metric)
    hubs = list(_state["hubs"])
    if min_athletes:
        hubs = [h for h in hubs if h.total_athletes >= min_athletes]
    if metric == "sport_count":
        hubs = [h for h in hubs if _hub_sport_count(h, sport) > 0]
    return sorted(
        hubs,
        key=lambda h: (
            -_hub_metric_value(h, metric, sport),
            -h.total_athletes,
            h.display_name,
        ),
    )


def _ranked_states(metric: str, min_athletes: int | None = None) -> list[StateAggregate]:
    metric = _normalize_metric(metric)
    states = list(_state["state_aggregates"])
    if metric == "paralympic_share":
        threshold = 25 if min_athletes is None else min_athletes
        states = [s for s in states if s.total_athletes >= threshold]
    elif min_athletes:
        states = [s for s in states if s.total_athletes >= min_athletes]
    return sorted(
        states,
        key=lambda s: (
            -_state_metric_value(s, metric),
            -s.total_athletes,
            _state_name(s.state),
        ),
    )


def _hub_rank(hub_id: str, metric: str, sport: str | None = None) -> tuple[int | None, int, Hub | None]:
    ranked = _ranked_hubs(metric, sport)
    hub = _state["hubs_by_id"].get(hub_id)
    for index, item in enumerate(ranked, 1):
        if item.hub_id == hub_id:
            return index, len(ranked), hub
    return None, len(ranked), hub


def _state_rank(
    state_code: str,
    metric: str,
    min_athletes: int | None = None,
) -> tuple[int | None, int, StateAggregate | None]:
    code = state_code.upper()
    ranked = _ranked_states(metric, min_athletes)
    agg = next((s for s in _state["state_aggregates"] if s.state == code), None)
    for index, item in enumerate(ranked, 1):
        if item.state == code:
            return index, len(ranked), agg
    return None, len(ranked), agg


def _hub_rank_bundle(hub: Hub) -> str:
    total_rank, total_n, _ = _hub_rank(hub.hub_id, "total_athletes")
    para_rank, para_n, _ = _hub_rank(hub.hub_id, "paralympic_athletes")
    share_rank, share_n, _ = _hub_rank(hub.hub_id, "paralympic_share")
    return (
        f"Ranks among hubs: total athletes #{total_rank} of {total_n}, "
        f"Paralympic athletes #{para_rank} of {para_n}, "
        f"Paralympic share #{share_rank} of {share_n}."
    )


def _state_rank_bundle(state: StateAggregate) -> str:
    total_rank, total_n, _ = _state_rank(state.state, "total_athletes")
    para_rank, para_n, _ = _state_rank(state.state, "paralympic_athletes")
    share_rank, share_n, _ = _state_rank(state.state, "paralympic_share")
    share_text = f"#{share_rank} of {share_n}" if share_rank else f"not ranked in the {share_n}-state 25+ athlete reliability set"
    return (
        f"Ranks among states/territories: total athletes #{total_rank} of {total_n}, "
        f"Paralympic athletes #{para_rank} of {para_n}, "
        f"Paralympic share {share_text}."
    )


def _ascii_search_text(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def _hometown_key(label: str, state_code: str | None = None) -> str:
    return f"{_ascii_search_text(label)}|{(state_code or '').upper()}"


def _session_id(value: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "", value or "")[:96]


def _prune_chat_sessions() -> None:
    now = time.time()
    expired = [
        sid for sid, data in _chat_sessions.items()
        if now - float(data.get("updated_at", 0)) > CHAT_SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _chat_sessions.pop(sid, None)


def _remember_session_turn(
    session_id: str | None,
    user_text: str,
    model_text: str,
    tool_calls: list[ChatToolCall] | list[dict[str, Any]] | None = None,
) -> None:
    sid = _session_id(session_id)
    if not sid:
        return
    _prune_chat_sessions()
    session = _chat_sessions.setdefault(sid, {"turns": [], "updated_at": time.time()})
    turns = session.setdefault("turns", [])
    normalized_tools: list[dict[str, Any]] = []
    for call in (tool_calls or [])[:4]:
        if isinstance(call, ChatToolCall):
            normalized_tools.append({"name": call.name, "args": call.args})
        elif isinstance(call, dict):
            normalized_tools.append({"name": call.get("name"), "args": call.get("args", {})})
    turns.append({
        "user": re.sub(r"\s+", " ", user_text or "").strip()[:700],
        "model": re.sub(r"\s+", " ", model_text or "").strip()[:900],
        "tools": normalized_tools,
    })
    del turns[:-CHAT_SESSION_MAX_TURNS]
    session["updated_at"] = time.time()


def _session_context_text(session_id: str | None) -> str:
    sid = _session_id(session_id)
    if not sid:
        return ""
    _prune_chat_sessions()
    session = _chat_sessions.get(sid)
    if not session:
        return ""
    lines: list[str] = []
    for turn in session.get("turns", [])[-6:]:
        if turn.get("user"):
            lines.append(f"User: {turn['user']}")
        if turn.get("model"):
            lines.append(f"Gemini: {turn['model']}")
        tools = turn.get("tools") or []
        if tools:
            lines.append("Tools: " + ", ".join(str(t.get("name")) for t in tools if t.get("name")))
    return "\n".join(lines)[-1800:]


def _build_hometown_index(raw_athletes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    buckets: dict[str, dict[str, Any]] = {}
    for athlete in raw_athletes:
        hometown = athlete.get("hometown") or {}
        label = str(hometown.get("label") or "").strip()
        lat = hometown.get("latitude")
        lon = hometown.get("longitude")
        if not label or lat is None or lon is None:
            continue
        parsed_lat = float(lat)
        parsed_lon = float(lon)
        state_code = state_from_latlon(parsed_lat, parsed_lon)
        key = _hometown_key(label, state_code)
        if key not in buckets:
            buckets[key] = {
                "label": label,
                "state": state_code,
                "lat_sum": 0.0,
                "lon_sum": 0.0,
                "total_athletes": 0,
                "olympic_count": 0,
                "paralympic_count": 0,
                "both_count": 0,
                "sports": Counter(),
                "hubs": Counter(),
                "distances": [],
            }
        bucket = buckets[key]
        bucket["lat_sum"] += parsed_lat
        bucket["lon_sum"] += parsed_lon
        bucket["total_athletes"] += 1
        status = athlete.get("status")
        if status == "olympic":
            bucket["olympic_count"] += 1
        elif status == "paralympic":
            bucket["paralympic_count"] += 1
        elif status == "both":
            bucket["both_count"] += 1
        for sport in athlete.get("sports") or []:
            if sport:
                bucket["sports"][str(sport)] += 1
        hub_id = athlete.get("hub_id")
        if hub_id:
            bucket["hubs"][str(hub_id)] += 1
        distance = athlete.get("distance_to_hub_km")
        if isinstance(distance, (int, float)):
            bucket["distances"].append(float(distance))

    hometowns: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, bucket in buckets.items():
        total = int(bucket["total_athletes"])
        hub_id = bucket["hubs"].most_common(1)[0][0] if bucket["hubs"] else ""
        hub = _state["hubs_by_id"].get(hub_id)
        para_total = int(bucket["paralympic_count"] + bucket["both_count"])
        item = {
            "hometown": bucket["label"],
            "state": bucket["state"],
            "lat": round(bucket["lat_sum"] / total, 6),
            "lon": round(bucket["lon_sum"] / total, 6),
            "total_athletes": total,
            "olympic_count": int(bucket["olympic_count"]),
            "paralympic_count": int(bucket["paralympic_count"]),
            "both_count": int(bucket["both_count"]),
            "paralympic_share": para_total / total if total else 0.0,
            "top_sports": [
                {"sport": _display_sport(sport), "count": count}
                for sport, count in bucket["sports"].most_common(5)
            ],
            "hub_id": hub_id,
            "hub_name": hub.display_name if hub else hub_id,
            "distance_to_hub_km": round(sum(bucket["distances"]) / len(bucket["distances"]), 1) if bucket["distances"] else None,
        }
        hometowns.append(item)
        by_key[key] = item
        by_name[_ascii_search_text(bucket["label"])].append(item)

    hometowns.sort(key=lambda item: (-item["total_athletes"], item["hometown"], item["state"]))
    for matches in by_name.values():
        matches.sort(key=lambda item: (-item["total_athletes"], item["state"], item["hometown"]))
    return hometowns, by_key, by_name


def _extract_state_from_text(value: str) -> tuple[str, str]:
    text = value or ""
    found_code = ""
    for code in sorted(STATE_CODE_TO_NAME, key=len, reverse=True):
        if re.search(rf"\b{re.escape(code)}\b", text, flags=re.IGNORECASE):
            found_code = code
            text = re.sub(rf"\b{re.escape(code)}\b", " ", text, flags=re.IGNORECASE)
            break
    if not found_code:
        lowered = text.lower()
        for name, code in sorted(STATE_NAME_TO_CODE.items(), key=lambda item: -len(item[0])):
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                found_code = code
                text = re.sub(rf"\b{re.escape(name)}\b", " ", text, flags=re.IGNORECASE)
                break
    text = re.sub(r"\b(usa|united states|hometown)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9 .'-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" ,."), found_code


def _extract_hometown_query(message: str) -> tuple[str, str]:
    cleaned = re.sub(r"\?", "", message or "").strip()
    patterns = [
        r"(?:from|in|near)\s+(?:my\s+)?hometown\s+(.+)$",
        r"(?:my\s+)?hometown\s+(?:is\s+)?(.+)$",
        r"(?:from|in|near)\s+([A-Za-z0-9 .,'-]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"\b(athletes?|mapped|team usa|are|there|come|came)\b", " ", candidate, flags=re.IGNORECASE)
            hometown, state_code = _extract_state_from_text(candidate)
            if hometown:
                return hometown, state_code
    return "", ""


def _hometown_matches(hometown: str, state_code: str | None = None) -> list[dict[str, Any]]:
    query = _ascii_search_text(hometown)
    if not query:
        return []
    exact = _state["hometowns_by_key"].get(_hometown_key(hometown, state_code or ""))
    if exact:
        return [exact]
    matches = list(_state["hometowns_by_name"].get(query, []))
    if state_code:
        matches = [m for m in matches if m.get("state") == state_code.upper()]
    if matches:
        return matches
    fuzzy = [
        item for item in _state["hometowns"]
        if query in _ascii_search_text(item["hometown"]) or _ascii_search_text(item["hometown"]) in query
    ]
    if state_code:
        fuzzy = [m for m in fuzzy if m.get("state") == state_code.upper()]
    return fuzzy[:8]


def _resolve_focus_hometown_args(args: dict[str, Any]) -> dict[str, Any]:
    raw_hometown = str(args.get("hometown") or args.get("query") or "").strip()
    explicit_state = str(args.get("state_code") or "").upper().strip()
    parsed_hometown, parsed_state = _extract_state_from_text(raw_hometown)
    hometown = parsed_hometown or raw_hometown
    state_code = explicit_state or parsed_state
    matches = _hometown_matches(hometown, state_code)
    if len(matches) == 1:
        match = dict(matches[0])
        match.update({
            "resolved": True,
            "ambiguous": False,
            "query": raw_hometown,
            "state_code": match.get("state"),
            "source": "dataset",
        })
        return match
    if len(matches) > 1:
        return {
            "resolved": False,
            "ambiguous": True,
            "query": raw_hometown,
            "hometown": hometown,
            "state_code": state_code,
            "geocode_query": ", ".join(part for part in [hometown, _state_name(state_code) if state_code else ""] if part),
            "options": [
                {
                    "hometown": m["hometown"],
                    "state": m["state"],
                    "total_athletes": m["total_athletes"],
                    "hub_id": m["hub_id"],
                    "hub_name": m["hub_name"],
                }
                for m in matches[:5]
            ],
            "source": "ambiguous",
        }
    return {
        "resolved": False,
        "ambiguous": False,
        "query": raw_hometown,
        "hometown": hometown or raw_hometown,
        "state_code": state_code,
        "total_athletes": 0,
        "olympic_count": 0,
        "paralympic_count": 0,
        "both_count": 0,
        "paralympic_share": 0,
        "top_sports": [],
        "hub_id": "",
        "hub_name": "",
        "geocode_query": ", ".join(part for part in [hometown or raw_hometown, _state_name(state_code) if state_code else ""] if part),
        "source": "geocode_fallback",
    }


def _prepare_tool_call_for_frontend(tool_name: str, args: dict[str, Any]) -> ChatToolCall:
    if tool_name == "focus_hometown":
        return ChatToolCall(name="focus_hometown", args=_resolve_focus_hometown_args(args))
    return ChatToolCall(name=tool_name, args=args)


def _hub_line(hub: Hub, metric: str, rank: int | None = None, sport: str | None = None) -> str:
    value = _format_metric_value(_hub_metric_value(hub, metric, sport), metric)
    para = _hub_para_count(hub)
    hot = " Hot Spot" if hub.is_paralympic_hot_spot else ""
    prefix = f"{rank}. " if rank else ""
    sport_text = f" for {_display_sport(sport)}" if metric == "sport_count" and sport else ""
    return (
        f"{prefix}{hub.display_name}{hot} - {value} {_metric_label(metric)}{sport_text}; "
        f"{hub.total_athletes} total athletes, {para} Paralympians, "
        f"{hub.composition.paralympic_share * 100:.1f}% Paralympic share"
    )


def _state_line(state: StateAggregate, metric: str, rank: int | None = None) -> str:
    value = _format_metric_value(_state_metric_value(state, metric), metric)
    para = _state_para_count(state)
    prefix = f"{rank}. " if rank else ""
    return (
        f"{prefix}{_state_name(state.state)} ({state.state}) - {value} {_metric_label(metric)}; "
        f"{state.total_athletes} total athletes, {para} Paralympians, "
        f"{state.paralympic_share * 100:.1f}% Paralympic share"
    )


def _resolve_state_codes_from_text(message: str) -> list[str]:
    normalized = f" {_normal_search_text(message)} "
    found: list[str] = []
    for name, code in STATE_NAME_TO_CODE.items():
        if f" {_normal_search_text(name)} " in normalized and code not in found:
            found.append(code)
    words = set(normalized.split())
    for code in STATE_CODE_TO_NAME:
        if code.lower() in words and code not in found:
            found.append(code)
    return found


def _extract_limit(message: str, default: int = 5) -> int:
    match = re.search(r"\btop\s+(\d+)\b|\brank\s+(?:the\s+)?(?:top\s+)?(\d+)\b", message.lower())
    if not match:
        return default
    return _limit(match.group(1) or match.group(2), default)


def _extract_sport(message: str) -> str:
    msg = message.lower()
    for marker in [" for ", " in ", " at "]:
        if marker in msg:
            candidate = msg.rsplit(marker, 1)[-1]
            candidate = re.sub(r"[^a-zA-Z -]", "", candidate).strip()
            if candidate and candidate not in {"hubs", "states", "share", "athletes"}:
                return candidate
    if "ski" in msg:
        return "skiing"
    if "swim" in msg:
        return "swimming"
    return ""


def _macro_region_from_text(message: str) -> str:
    normalized = _normal_search_text(message)
    for region in MACRO_REGIONS:
        if _normal_search_text(region) in normalized:
            return region
    if "mountain" in normalized:
        return "Mountain West"
    if "mid atlantic" in normalized or "midatlantic" in normalized:
        return "Mid-Atlantic"
    return ""


def _is_analyst_request(message: str) -> bool:
    msg = message.lower()
    analyst_terms = [
        "rank", "ranking", "compare", "comparison", " versus ", " vs ",
        "top ", "leading", "most", "least", "which", "how many",
        "total", "overall", "share", "percentage", "strongest", "list",
    ]
    return any(term in msg for term in analyst_terms)


def _is_map_explain_request(message: str) -> bool:
    msg = _normal_search_text(message)
    explain_terms = [
        "what do the dots mean", "what do dots mean", "little dots",
        "red dots", "blue dots", "red circles", "blue circles",
        "what are the dots", "what are the circles", "legend",
        "how do i read", "read the map", "what does red mean",
        "what does blue mean", "territories", "insets",
        "alaska inset", "hawaii inset", "puerto rico inset",
        "state shading", "colors mean",
        "what are hot spots", "what is a hot spot", "what do hot spots mean",
        "what is this project", "what is this map", "what is this tool",
        "what is this about", "project about", "explain this for judges",
        "explain this project", "explain the project", "demo this",
    ]
    return any(term in msg for term in explain_terms)


def _is_hometown_lookup_request(message: str) -> bool:
    msg = _normal_search_text(message)
    return (
        "hometown" in msg
        or ("how many" in msg and (" from " in f" {message.lower()} " or " in " in f" {message.lower()} "))
        or ("athletes from" in msg)
    )


def _direct_query_tool_call(message: str) -> ChatToolCall | None:
    msg = message.lower()
    metric = _metric_from_text(message)
    limit = _extract_limit(message)
    state_codes = _resolve_state_codes_from_text(message)
    hub = _resolve_hub_from_message(message)
    macro_region = _macro_region_from_text(message)

    if _is_map_explain_request(message):
        topic = ""
        for candidate in ["dots", "circles", "red", "blue", "territories", "legend", "hot spots", "state shading"]:
            if candidate.replace(" ", "") in msg.replace(" ", ""):
                topic = candidate
                break
        return ChatToolCall(name="explain_map", args={"topic": topic})

    if _is_hometown_lookup_request(message):
        hometown, state_code = _extract_hometown_query(message)
        if hometown:
            return _prepare_tool_call_for_frontend(
                "focus_hometown",
                {"hometown": hometown, **({"state_code": state_code} if state_code else {})},
            )

    if "how many" in msg or "summary" in msg or ("athlete" in msg and "hub" in msg):
        return ChatToolCall(name="query_data", args={"query_type": "summary"})

    if macro_region and ("hub" in msg or "region" in msg or "athlete" in msg):
        return ChatToolCall(
            name="query_data",
            args={
                "query_type": "hubs_by_macro_region",
                "macro_region": macro_region,
                "limit": limit,
            },
        )

    if "compare" in msg or " vs " in msg or " versus " in msg:
        if len(state_codes) >= 2:
            return ChatToolCall(
                name="query_data",
                args={"query_type": "compare_states", "state_codes": state_codes[:4]},
            )
        hubs = []
        for candidate in _state["hubs"]:
            if candidate.hub_id == (hub.hub_id if hub else ""):
                hubs.append(candidate.hub_id)
                continue
            candidate_text = " ".join([
                candidate.display_name,
                candidate.medoid_hometown,
                candidate.region_name,
                *candidate.search_aliases,
            ])
            if _normal_search_text(candidate_text) in _normal_search_text(message):
                hubs.append(candidate.hub_id)
        if len(hubs) >= 2:
            return ChatToolCall(
                name="query_data",
                args={"query_type": "compare_hubs", "hub_ids": hubs[:4]},
            )

    if "strongest" in msg or ("top" in msg and ("sport" in msg or "ski" in msg or "swim" in msg)):
        sport = _extract_sport(message)
        return ChatToolCall(
            name="query_data",
            args={"query_type": "hubs_by_sport", "sport": sport, "limit": limit},
        )

    if "what rank" in msg or "rank is" in msg or "rank does" in msg:
        if hub:
            return ChatToolCall(
                name="query_data",
                args={
                    "query_type": "entity_rank",
                    "entity_type": "hub",
                    "hub_id": hub.hub_id,
                    "metric": metric,
                },
            )
        if state_codes:
            return ChatToolCall(
                name="query_data",
                args={
                    "query_type": "entity_rank",
                    "entity_type": "state",
                    "state_code": state_codes[0],
                    "metric": metric,
                },
            )

    if "rank" in msg or "top" in msg or "list" in msg:
        entity_type = "state" if "state" in msg or "states" in msg else "hub"
        return ChatToolCall(
            name="query_data",
            args={
                "query_type": "rank_list",
                "entity_type": entity_type,
                "metric": metric,
                "limit": limit,
                **({"sport": _extract_sport(message)} if metric == "sport_count" else {}),
            },
        )

    return None


def _normal_search_text(value: str) -> str:
    return _ascii_search_text(value)


def _resolve_hub_from_message(message: str) -> Hub | None:
    needle = _normal_search_text(message)
    if not needle:
        return None
    matches: list[tuple[Hub, int]] = []
    for hub in _state["hubs"]:
        candidates = [
            hub.hub_id,
            hub.display_name,
            hub.medoid_hometown,
            hub.region_name,
            *hub.search_aliases,
        ]
        for candidate in candidates:
            normalized = _normal_search_text(candidate)
            if not normalized or len(normalized) <= 2:
                continue
            if normalized in needle:
                matches.append((hub, len(normalized)))
                break

    if not matches:
        return None
    return max(matches, key=lambda item: item[1])[0]


def _direct_chat_response(req: ChatRequest) -> ChatResponse | None:
    msg = req.message.lower()
    tool_call: ChatToolCall | None = None

    if any(term in msg for term in ["reset", "clear view", "start over", "national view"]):
        tool_call = ChatToolCall(name="reset_view", args={})
    elif (query_call := _direct_query_tool_call(req.message)) is not None:
        tool_call = query_call
    elif "hot spot" in msg or "hotspot" in msg:
        tool_call = ChatToolCall(name="filter_to_paralympic", args={})
    elif not _is_analyst_request(req.message):
        hub = _resolve_hub_from_message(req.message)
        if hub:
            tool_call = ChatToolCall(name="select_hub", args={"hub_id": hub.hub_id})

    if not tool_call:
        return None

    reply_text = _build_tool_result_context(tool_call.name, tool_call.args)
    new_history = list(req.history)
    new_history.append({"role": "user", "text": req.message})
    new_history.append({"role": "model", "text": reply_text})
    _remember_session_turn(req.session_id, req.message, reply_text, [tool_call])
    return ChatResponse(text=reply_text, tool_calls=[tool_call], history=new_history)


def _load_data() -> None:
    # Try multiple possible roots so this works locally AND in Cloud Run.
    # Local: backend/main.py -> parent.parent is project root
    # Cloud Run: /app/main.py -> parent is /app, which has pipeline/ copied alongside
    candidate_roots = [
        Path(__file__).parent.parent,  # local dev
        Path(__file__).parent,         # Cloud Run /app
        Path("/app"),                  # absolute fallback
    ]

    hubs_path = None
    narratives_path = None
    athletes_path = None

    for root in candidate_roots:
        candidate_hubs = root / "pipeline" / "clustered" / "hubs.json"
        candidate_narratives = root / "pipeline" / "narratives" / "hubs.json"
        candidate_athletes = root / "pipeline" / "clustered" / "athletes.json"
        if candidate_hubs.exists() and candidate_narratives.exists() and candidate_athletes.exists():
            hubs_path = candidate_hubs
            narratives_path = candidate_narratives
            athletes_path = candidate_athletes
            logger.info(f"Loading data from: {root}")
            break

    if hubs_path is None:
        # No location worked. Show what we tried so the error log is actionable.
        tried = [str(r) for r in candidate_roots]
        raise RuntimeError(
            f"Data files not found in any candidate location: {tried}. "
            f"Looked for pipeline/clustered/hubs.json, pipeline/narratives/hubs.json, pipeline/clustered/athletes.json."
        )
    if not hubs_path.exists():
        raise RuntimeError(f"Hubs file not found: {hubs_path}")
    if not narratives_path.exists():
        raise RuntimeError(f"Narratives file not found: {narratives_path}")
    if not athletes_path.exists():
        raise RuntimeError(f"Athletes file not found: {athletes_path}")

    with hubs_path.open("r", encoding="utf-8") as f:
        raw_hubs = json.load(f)
    hubs = [Hub.model_validate(h) for h in raw_hubs]

    with narratives_path.open("r", encoding="utf-8") as f:
        raw_narratives = json.load(f)
    narratives = {
        hub_id: HubNarrative.model_validate(n)
        for hub_id, n in raw_narratives.items()
    }

    with athletes_path.open("r", encoding="utf-8") as f:
        raw_athletes = json.load(f)
    
    athletes_geo_points = []
    state_counts = defaultdict(lambda: {"olympic": 0, "paralympic": 0, "both": 0})

    for a in raw_athletes:
        hometown = a.get("hometown", {})
        lat = hometown.get("latitude")
        lon = hometown.get("longitude")
        
        if lat is not None and lon is not None:
            parsed_lat = float(lat)
            parsed_lon = float(lon)
            st = state_from_latlon(parsed_lat, parsed_lon)
            status = a.get("status")

            athletes_geo_points.append(
                AthleteGeoPoint(
                    hub_id=a.get("hub_id", "UNKNOWN"),
                    lat=round(parsed_lat, 4),
                    lon=round(parsed_lon, 4),
                    status=status,
                    state=st
                )
            )

            if st != "XX" and status in ("olympic", "paralympic", "both"):
                state_counts[st][status] += 1

    state_aggregates = []
    for st, counts in state_counts.items():
        oly = counts["olympic"]
        para = counts["paralympic"]
        both = counts["both"]
        total = oly + para + both
        if total > 0:
            para_share = (para + both) / total
            state_aggregates.append(
                StateAggregate(
                    state=st,
                    total_athletes=total,
                    olympic_count=oly,
                    paralympic_count=para,
                    both_count=both,
                    paralympic_share=para_share
                )
            )
    
    state_aggregates.sort(key=lambda x: x.total_athletes, reverse=True)

    _state["hubs"] = hubs
    _state["hubs_by_id"] = {h.hub_id: h for h in hubs}
    _state["narratives"] = narratives
    _state["athletes_geo_points"] = athletes_geo_points
    _state["state_aggregates"] = state_aggregates
    hometowns, hometowns_by_key, hometowns_by_name = _build_hometown_index(raw_athletes)
    _state["hometowns"] = hometowns
    _state["hometowns_by_key"] = hometowns_by_key
    _state["hometowns_by_name"] = hometowns_by_name

    logger.info(
        f"Loaded {len(hubs)} hubs, {len(narratives)} narratives, {len(athletes_geo_points)} athletes. "
        f"Aggregated {len(state_aggregates)} states. "
        f"{sum(1 for h in hubs if h.is_paralympic_hot_spot)} Paralympic Hot Spots. "
        f"Indexed {len(hometowns)} hometown aggregates."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_data()
    yield


app = FastAPI(
    title="Hometown Success Engine API",
    description="Team USA hometown hub data with regional context and Gemini map tools",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hometown-success-engine-11a06.web.app",
        "https://hometown-success-engine-11a06.firebaseapp.com",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "hubs_loaded": len(_state["hubs"]),
        "narratives_loaded": len(_state["narratives"]),
        "athletes_loaded": len(_state["athletes_geo_points"]),
        "states_with_athletes": len(_state["state_aggregates"]),
    }


@app.get("/hubs", response_model=list[Hub])
def list_hubs() -> list[Hub]:
    return _state["hubs"]


@app.get("/hubs/{hub_id}", response_model=Hub)
def get_hub(hub_id: str) -> Hub:
    hub = _state["hubs_by_id"].get(hub_id)
    if hub is None:
        raise HTTPException(404, f"Hub {hub_id} not found")
    return hub


@app.get("/hubs/{hub_id}/narrative", response_model=HubNarrative)
def get_hub_narrative(hub_id: str) -> HubNarrative:
    narrative = _state["narratives"].get(hub_id)
    if narrative is None:
        raise HTTPException(404, f"Narrative for {hub_id} not found")
    return narrative


@app.get("/athletes", response_model=list[AthleteGeoPoint])
def list_athletes() -> list[AthleteGeoPoint]:
    return _state["athletes_geo_points"]


@app.get("/states/aggregate", response_model=list[StateAggregate])
def list_state_aggregates() -> list[StateAggregate]:
    return _state["state_aggregates"]


@app.post("/tools/filter_map", response_model=SetFilterAction)
def tool_filter_map(req: FilterMapRequest) -> SetFilterAction:
    filter_update = FilterUpdate(
        macro_region=req.macro_region,
        region_name=req.region_name,
        paralympic_focus=req.paralympic_focus,
        sport_category=req.sport_category,
    )
    return SetFilterAction(filter=filter_update)


@app.post("/tools/zoom_to_hub", response_model=SelectHubAction)
def tool_zoom_to_hub(req: ZoomToHubRequest) -> SelectHubAction:
    if req.hub_id not in _state["hubs_by_id"]:
        raise HTTPException(404, f"Hub {req.hub_id} not found")
    return SelectHubAction(hub_id=req.hub_id)

def _build_tool_result_context(tool_name: str, args: dict) -> str:
    """Build a fact-rich tool-result string that Gemini will use to narrate
    what happened. The richer this string, the better the narration."""

    stats = _dataset_stats()
    baseline = stats["baseline_pct"]
    baseline_text = f"{baseline:.1f}%"
    hot_spot_threshold_text = f"{PARALYMPIC_HOT_SPOT_THRESHOLD_PCT:.1f}%"
    hot_spot_count = stats["hot_spot_count"]

    if tool_name == "explain_map":
        topic = _normal_search_text(str(args.get("topic") or "legend"))
        base = (
            f"Map guide: the blue state shading shows mapped athlete density by state or territory, "
            f"with darker blue meaning more mapped athletes. Large circles are the 40 hometown hubs; "
            f"blue circles are standard hubs and red circles are Paralympic Hot Spots at or above the "
            f"{hot_spot_threshold_text} Paralympic-share threshold. Small constellation dots are individual "
            f"mapped athlete hometown points: blue dots are Olympians and red dots are Paralympians or athletes "
            f"tagged as both. Alaska, Hawaii, and Puerto Rico appear as insets so their geography stays visible "
            f"alongside the continental map."
        )
        if "red" in topic:
            return base + f" In short: red means Paralympic focus, either a small Paralympian dot or a larger Hot Spot hub."
        if "blue" in topic:
            return base + " In short: blue means general Olympic/hub density context, either state shading, standard hubs, or Olympian dots."
        if "territor" in topic or "inset" in topic:
            return base + " The inset row keeps non-contiguous Team USA geographies visible without distorting the main map."
        return base

    if tool_name == "focus_hometown":
        focus = _resolve_focus_hometown_args(args)
        requested = focus.get("query") or focus.get("hometown") or "that hometown"
        if focus.get("ambiguous"):
            options = focus.get("options") or []
            option_text = "; ".join(
                f"{o['hometown']}, {o['state']} ({o['total_athletes']} mapped athletes, closest hub {o['hub_name']})"
                for o in options
            )
            return (
                f"I found multiple mapped hometown matches for {requested}. "
                f"Please include a state for a deterministic lookup. Options: {option_text}."
            )
        if not focus.get("resolved"):
            geocode_query = focus.get("geocode_query") or requested
            return (
                f"No exact mapped hometown match was found for {requested}. "
                f"The map can still zoom to {geocode_query} using Google Maps geocoding, but the current dataset has "
                f"0 mapped athletes for that exact hometown label. Use the nearest hub for regional context."
            )
        total = int(focus.get("total_athletes") or 0)
        para = int(focus.get("paralympic_count") or 0) + int(focus.get("both_count") or 0)
        para_share = float(focus.get("paralympic_share") or 0) * 100
        top_sports = ", ".join(
            f"{item['sport']} ({item['count']})"
            for item in (focus.get("top_sports") or [])[:3]
        ) or "various sports"
        hub_id = focus.get("hub_id") or ""
        hub = _state["hubs_by_id"].get(hub_id)
        hub_context = ""
        if hub:
            hub_context = (
                f" Assigned hub: {hub.display_name}, which has {hub.total_athletes} mapped athletes "
                f"and {hub.composition.paralympic_share * 100:.1f}% Paralympic share."
            )
        return (
            f"Hometown focus: {focus['hometown']}, {focus['state']} has {total} mapped athletes in the public dataset: "
            f"{focus['olympic_count']} Olympians, {para} Paralympians, {para_share:.1f}% Paralympic share. "
            f"Top sports from this hometown: {top_sports}.{hub_context}"
        )

    if tool_name == "filter_to_paralympic":
        macro_region = args.get("macro_region")
        hot_spots = [h for h in _state["hubs"] if h.is_paralympic_hot_spot]
        if macro_region:
            filtered = [h for h in hot_spots if h.macro_region == macro_region]
            if not filtered:
                return (
                    f"The Paralympic filter was applied for the {macro_region} region, "
                    f"but no hubs there meet the {hot_spot_threshold_text} Hot Spot threshold. The {hot_spot_count} national Hot Spots "
                    f"are in: " + ", ".join(f"{h.display_name}" for h in hot_spots) + "."
                )
            top = max(filtered, key=lambda h: h.composition.paralympic_share)
            return (
                f"Filter applied: highlighting Paralympic Hot Spots in {macro_region}. "
                f"Hubs visible: {', '.join(h.display_name for h in filtered)}. "
                f"Leading: {top.display_name} at {top.composition.paralympic_share*100:.1f}% Paralympic, "
                f"{top.total_athletes} athletes total. Hot Spot threshold: {hot_spot_threshold_text}; national Paralympic baseline: {baseline_text}."
            )
        else:
            sorted_hot = sorted(hot_spots, key=lambda h: -h.composition.paralympic_share)
            top = sorted_hot[0]
            return (
                f"Filter applied: highlighting all {len(sorted_hot)} Paralympic Hot Spots, hubs where "
                f"Paralympic share is at or above the {hot_spot_threshold_text} Hot Spot threshold. "
                f"Top spot: {top.display_name} at {top.composition.paralympic_share*100:.1f}% Paralympic "
                f"({top.total_athletes} athletes). All {len(sorted_hot)}: " +
                ", ".join(
                    f"{h.display_name} ({h.composition.paralympic_share*100:.1f}%)"
                    for h in sorted_hot
                ) + "."
            )

    if tool_name == "select_hub" or tool_name == "zoom_to_hub":
        hub_id = args.get("hub_id", "")
        hub = _state["hubs_by_id"].get(hub_id)
        if not hub:
            return f"Action {tool_name} attempted but hub {hub_id} not found."
        narrative = _state["narratives"].get(hub_id)
        para_pct = hub.composition.paralympic_share * 100
        top_sport = _display_sport(hub.top_sports[0].sport) if hub.top_sports else "various sports"
        top_sports = ", ".join(f"{_display_sport(sp.sport)} ({sp.count})" for sp in hub.top_sports[:3]) or "various sports"
        result = (
            f"Map zoomed/selected: {hub.display_name} ({hub.region_name}, {hub.macro_region}). "
            f"{hub.total_athletes} athletes total, {hub.composition.olympic_count} Olympians, "
            f"{_hub_para_count(hub)} Paralympians, {para_pct:.1f}% Paralympic share. "
            f"Top sport: {top_sport}. Top sports: {top_sports}. "
            f"{_hub_rank_bundle(hub)} "
            f"{'This is a Paralympic Hot Spot.' if hub.is_paralympic_hot_spot else 'Not a Paralympic Hot Spot.'}"
        )
        if narrative:
            result += f" Narrative headline: '{narrative.headline}'."
            if narrative.paralympic_callout:
                result += f" Paralympic callout: '{narrative.paralympic_callout}'."
            if narrative.climate:
                c = narrative.climate
                climate_parts = []
                if c.annual_avg_temp_f is not None:
                    climate_parts.append(f"avg {c.annual_avg_temp_f}°F")
                if c.annual_precipitation_in is not None:
                    climate_parts.append(f"{c.annual_precipitation_in}in precip/yr")
                if c.elevation_ft is not None:
                    climate_parts.append(f"{int(c.elevation_ft)}ft elevation")
                if climate_parts:
                    result += f" Climate: {', '.join(climate_parts)}."
            if narrative.geographic_context:
                result += f" Geographic context: {narrative.geographic_context}"
        return result

    if tool_name == "select_state":
            code = args.get("state_code", "").upper()
            name = STATE_CODE_TO_NAME.get(code, code)
            agg = next((s for s in _state["state_aggregates"] if s.state == code), None)
            hubs_in_state = [h for h in _state["hubs"] if code in h.states]
            top_hub = max(hubs_in_state, key=lambda h: h.total_athletes) if hubs_in_state else None
            if not agg:
                base = (
                    f"State panel opened: {name} ({code}). "
                    f"This state has 0 athletes mapped in our 2020 to 2026 dataset, "
                    f"so it ranks last across our 52 mappable regions. "
                    f"The {hot_spot_count} Paralympic Hot Spots are all elsewhere."
                )
                if top_hub:
                    base += (
                        f" One nearby hub does cover {name}: {top_hub.display_name} "
                        f"with {top_hub.total_athletes} athletes."
                    )
                return base
            para = agg.paralympic_count + agg.both_count
            para_pct = agg.paralympic_share * 100
            sorted_total = sorted(_state["state_aggregates"], key=lambda s: -s.total_athletes)
            total_rank = next((i + 1 for i, s in enumerate(sorted_total) if s.state == code), None)
            sorted_para = sorted(_state["state_aggregates"], key=lambda s: -(s.paralympic_count + s.both_count))
            para_rank = next((i + 1 for i, s in enumerate(sorted_para) if s.state == code), None)
            above_or_below = "above" if para_pct > baseline else "below"
            parts = [
                f"State panel opened: {name} ({code}).",
                f"{agg.total_athletes} athletes total, {para} Paralympians, {para_pct:.1f}% Paralympic share.",
                f"This is {above_or_below} the {baseline_text} national baseline.",
                f"Total athletes rank: #{total_rank}. Paralympic athlete rank: #{para_rank}.",
                _state_rank_bundle(agg),
            ]
            if top_hub:
                hot_tag = ", a Paralympic Hot Spot" if top_hub.is_paralympic_hot_spot else ""
                parts.append(
                    f"Top hub in {name}: {top_hub.display_name} with {top_hub.total_athletes} athletes "
                    f"({top_hub.composition.paralympic_share*100:.1f}% Paralympic{hot_tag})."
                )
            return " ".join(parts)

    if tool_name == "query_data":
        query_type = args.get("query_type", "")
        limit = min(int(args.get("limit", 5) or 5), 20)
        sport = (args.get("sport") or "").lower().strip()
        macro_region = (args.get("macro_region") or "").strip()
        entity_type = (args.get("entity_type") or "").lower().strip()
        metric = _normalize_metric(args.get("metric"), "total_athletes")
        state_code = (args.get("state_code") or "").upper().strip()
        hub_id = (args.get("hub_id") or "").strip()
        state_codes = [
            str(code).upper()
            for code in (args.get("state_codes") or [])
            if str(code).upper() in STATE_CODE_TO_NAME
        ]
        hub_ids = [
            str(hid)
            for hid in (args.get("hub_ids") or [])
            if str(hid) in _state["hubs_by_id"]
        ]
        min_athletes = args.get("min_athletes")
        try:
            min_athletes = int(min_athletes) if min_athletes is not None else None
        except (TypeError, ValueError):
            min_athletes = None

        aggregates = _state["state_aggregates"]
        hubs = _state["hubs"]

        legacy_aliases = {
            "top_states_by_total": {"query_type": "rank_list", "entity_type": "state", "metric": "total_athletes"},
            "top_states_by_paralympic": {"query_type": "rank_list", "entity_type": "state", "metric": "paralympic_athletes"},
            "top_states_by_paralympic_share": {"query_type": "rank_list", "entity_type": "state", "metric": "paralympic_share"},
            "top_hubs_by_total": {"query_type": "rank_list", "entity_type": "hub", "metric": "total_athletes"},
            "top_hubs_by_paralympic_share": {"query_type": "rank_list", "entity_type": "hub", "metric": "paralympic_share"},
        }
        if query_type in legacy_aliases:
            legacy = legacy_aliases[query_type]
            query_type = legacy["query_type"]
            entity_type = legacy["entity_type"]
            metric = legacy["metric"]

        if query_type == "rank_list":
            if entity_type == "state":
                ranked_states = _ranked_states(metric, min_athletes)[:limit]
                qualifier = ""
                if metric == "paralympic_share":
                    qualifier = " among states/territories with at least 25 mapped athletes"
                lines = [f"Top {len(ranked_states)} states/territories by {_metric_label(metric)}{qualifier}:"]
                for i, state in enumerate(ranked_states, 1):
                    lines.append(_state_line(state, metric, i))
                return " ".join(lines)

            ranked_hubs = _ranked_hubs(metric, sport, min_athletes)[:limit]
            sport_text = f" for {_display_sport(sport)}" if metric == "sport_count" and sport else ""
            lines = [f"Top {len(ranked_hubs)} hubs by {_metric_label(metric)}{sport_text}:"]
            for i, hub in enumerate(ranked_hubs, 1):
                lines.append(_hub_line(hub, metric, i, sport))
            return " ".join(lines)

        if query_type == "entity_rank":
            if entity_type == "state" or state_code:
                if not state_code:
                    return "No state was specified for the ranking lookup."
                rank, universe, agg = _state_rank(state_code, metric, min_athletes)
                if not agg:
                    return f"{_state_name(state_code)} ({state_code}) is not present in the mapped athlete dataset."
                value = _format_metric_value(_state_metric_value(agg, metric), metric)
                if rank is None:
                    return (
                        f"{_state_name(state_code)} ({state_code}) is outside the current {_metric_label(metric)} ranking universe "
                        f"of {universe} states/territories because it does not meet the minimum-athlete threshold. "
                        f"It has {agg.total_athletes} mapped athletes and {agg.paralympic_share * 100:.1f}% Paralympic share."
                    )
                baseline_note = ""
                if metric == "paralympic_share":
                    relation = "above" if agg.paralympic_share * 100 > baseline else "below"
                    baseline_note = f" This is {relation} the {baseline_text} national baseline."
                return (
                    f"{_state_name(state_code)} ({state_code}) ranks #{rank} of {universe} states/territories by "
                    f"{_metric_label(metric)} with {value}.{baseline_note} "
                    f"{_state_line(agg, metric)}"
                )

            if not hub_id:
                return "No hub was specified for the ranking lookup."
            rank, universe, hub = _hub_rank(hub_id, metric, sport)
            if not hub:
                return f"Hub {hub_id} is not present in the current map dataset."
            value = _format_metric_value(_hub_metric_value(hub, metric, sport), metric)
            baseline_note = ""
            if metric == "paralympic_share":
                relation = "above" if hub.composition.paralympic_share * 100 > baseline else "below"
                baseline_note = f" This is {relation} the {baseline_text} national baseline."
            hot_note = " It is a Paralympic Hot Spot." if hub.is_paralympic_hot_spot else " It is not a Paralympic Hot Spot."
            return (
                f"{hub.display_name} ranks #{rank} of {universe} hubs by {_metric_label(metric)} with {value}."
                f"{baseline_note}{hot_note} {_hub_line(hub, metric, None, sport)}"
            )

        if query_type == "state_profile":
            if not state_code:
                return "No state was specified for the state profile."
            agg = next((s for s in aggregates if s.state == state_code), None)
            if not agg:
                return f"{_state_name(state_code)} ({state_code}) has no mapped athletes in this dataset."
            top_hub = max(
                [h for h in hubs if state_code in h.states],
                key=lambda h: h.total_athletes,
                default=None,
            )
            result = f"State profile: {_state_line(agg, 'total_athletes')} {_state_rank_bundle(agg)}"
            if top_hub:
                result += f" Leading hub: {top_hub.display_name}, {top_hub.total_athletes} athletes, {top_hub.composition.paralympic_share * 100:.1f}% Paralympic share."
            return result

        if query_type == "hub_profile":
            if not hub_id:
                return "No hub was specified for the hub profile."
            hub = _state["hubs_by_id"].get(hub_id)
            if not hub:
                return f"Hub {hub_id} is not present in the current map dataset."
            return _build_tool_result_context("select_hub", {"hub_id": hub_id})

        if query_type == "compare_states":
            if len(state_codes) < 2:
                return "At least two states are required for a state comparison."
            parts = ["State comparison:"]
            for code in state_codes[:4]:
                agg = next((s for s in aggregates if s.state == code), None)
                if agg:
                    parts.append(_state_line(agg, "total_athletes"))
                    parts.append(_state_rank_bundle(agg))
            return " ".join(parts)

        if query_type == "compare_hubs":
            if len(hub_ids) < 2:
                return "At least two hubs are required for a hub comparison."
            parts = ["Hub comparison:"]
            for hid in hub_ids[:4]:
                hub = _state["hubs_by_id"].get(hid)
                if hub:
                    parts.append(_hub_line(hub, "total_athletes"))
                    parts.append(_hub_rank_bundle(hub))
                    if hub.top_sports:
                        parts.append("Top sports: " + ", ".join(f"{_display_sport(sp.sport)} ({sp.count})" for sp in hub.top_sports[:3]) + ".")
            return " ".join(parts)

        if query_type == "top_states_by_total":
            sorted_states = sorted(aggregates, key=lambda s: -s.total_athletes)[:limit]
            lines = [f"Top {limit} states by total athletes:"]
            for i, s in enumerate(sorted_states, 1):
                name = STATE_CODE_TO_NAME.get(s.state, s.state)
                para = s.paralympic_count + s.both_count
                lines.append(f"{i}. {name} ({s.state}) - {s.total_athletes} total, {para} Paralympians, {s.paralympic_share*100:.1f}% Para share")
            return " ".join(lines)

        if query_type == "top_states_by_paralympic":
            sorted_states = sorted(aggregates, key=lambda s: -(s.paralympic_count + s.both_count))[:limit]
            lines = [f"Top {limit} states by Paralympic athlete count:"]
            for i, s in enumerate(sorted_states, 1):
                name = STATE_CODE_TO_NAME.get(s.state, s.state)
                para = s.paralympic_count + s.both_count
                lines.append(f"{i}. {name} ({s.state}) - {para} Paralympians out of {s.total_athletes} total ({s.paralympic_share*100:.1f}%)")
            return " ".join(lines)

        if query_type == "top_states_by_paralympic_share":
            qualified = [s for s in aggregates if s.total_athletes >= 25]
            sorted_states = sorted(qualified, key=lambda s: -s.paralympic_share)[:limit]
            lines = [f"Top {limit} states by Paralympic share (states with 25+ athletes for statistical reliability, national baseline {baseline_text}):"]
            for i, s in enumerate(sorted_states, 1):
                name = STATE_CODE_TO_NAME.get(s.state, s.state)
                para = s.paralympic_count + s.both_count
                lines.append(f"{i}. {name} ({s.state}) - {s.paralympic_share*100:.1f}% Paralympic ({para} of {s.total_athletes} athletes)")
            return " ".join(lines)

        if query_type == "top_hubs_by_total":
            sorted_hubs = sorted(hubs, key=lambda h: -h.total_athletes)[:limit]
            lines = [f"Top {limit} hubs by total athletes:"]
            for i, h in enumerate(sorted_hubs, 1):
                hot = " (Hot Spot)" if h.is_paralympic_hot_spot else ""
                lines.append(f"{i}. {h.display_name}{hot} - {h.total_athletes} athletes, {h.composition.paralympic_share*100:.1f}% Paralympic")
            return " ".join(lines)

        if query_type == "top_hubs_by_paralympic_share":
            sorted_hubs = sorted(hubs, key=lambda h: -h.composition.paralympic_share)[:limit]
            lines = [f"Top {limit} hubs by Paralympic share (national baseline {baseline_text}):"]
            for i, h in enumerate(sorted_hubs, 1):
                hot = " (Hot Spot)" if h.is_paralympic_hot_spot else ""
                lines.append(f"{i}. {h.display_name}{hot} - {h.composition.paralympic_share*100:.1f}% Paralympic, {h.total_athletes} athletes")
            return " ".join(lines)

        if query_type == "all_hot_spots":
            hot_spots = sorted([h for h in hubs if h.is_paralympic_hot_spot], key=lambda h: -h.composition.paralympic_share)
            lines = [f"All {len(hot_spots)} Paralympic Hot Spots (hubs at or above the {hot_spot_threshold_text} Paralympic-share threshold; national baseline {baseline_text}):"]
            for i, h in enumerate(hot_spots, 1):
                top_sport = _display_sport(h.top_sports[0].sport) if h.top_sports else "various sports"
                lines.append(f"{i}. {h.display_name} - {h.composition.paralympic_share*100:.1f}% Paralympic, {h.total_athletes} athletes, top sport: {top_sport}")
            return " ".join(lines)

        if query_type == "hubs_by_sport":
            if not sport:
                return "No sport specified. Provide a sport name like 'swimming' or 'wheelchair basketball'."
            matches = []
            for h in hubs:
                for sp in h.top_sports:
                    if sport in sp.sport.lower():
                        matches.append((h, sp))
                        break
            matches.sort(key=lambda x: -x[1].count)
            matches = matches[:limit]
            if not matches:
                return f"No hubs found with '{sport}' as a top sport across our {len(hubs)} hubs."
            lines = [f"Top {len(matches)} hubs where '{sport}' appears among top sports:"]
            for i, (h, sp) in enumerate(matches, 1):
                lines.append(f"{i}. {h.display_name} - {sp.count} {_display_sport(sp.sport)} athletes ({sp.paralympic_count} Paralympic), hub total {h.total_athletes}")
            return " ".join(lines)

        if query_type == "hubs_by_macro_region":
            if not macro_region:
                return "No macro region specified."
            matches = [h for h in hubs if h.macro_region.lower() == macro_region.lower()]
            matches.sort(key=lambda h: -h.total_athletes)
            matches = matches[:limit]
            if not matches:
                regions_available = sorted(set(h.macro_region for h in hubs))
                return f"No hubs found in macro region '{macro_region}'. Available regions: {', '.join(regions_available)}."
            lines = [f"Hubs in {macro_region} ({len(matches)} shown):"]
            for i, h in enumerate(matches, 1):
                hot = " (Hot Spot)" if h.is_paralympic_hot_spot else ""
                lines.append(f"{i}. {h.display_name}{hot} - {h.total_athletes} athletes, {h.composition.paralympic_share*100:.1f}% Paralympic")
            return " ".join(lines)

        if query_type == "summary":
            total_athletes = sum(h.total_athletes for h in hubs)
            total_para = sum(h.composition.paralympic_count + h.composition.both_count for h in hubs)
            hot_spots = sum(1 for h in hubs if h.is_paralympic_hot_spot)
            states_count = len(aggregates)
            overall_para_pct = (total_para / total_athletes * 100) if total_athletes else 0
            return (
                f"Dataset summary: {total_athletes:,} Olympians and Paralympians from the 2020 to 2026 cycle, "
                f"mapped across {len(hubs)} hometown hubs and {states_count} US states and territories. "
                f"{hot_spots} hubs qualify as Paralympic Hot Spots at or above the {hot_spot_threshold_text} Paralympic-share threshold. "
                f"Overall Paralympic share across the dataset: {overall_para_pct:.1f}%."
            )

        return f"Unknown query_type: {query_type}"

    if tool_name == "reset_view":
        return (
            "Map reset to default continental US view. All filters cleared. "
            f"Total athletes mapped: {sum(h.total_athletes for h in _state['hubs']):,}. "
            f"Total hubs: {len(_state['hubs'])}. "
            f"Total Paralympic Hot Spots: {sum(1 for h in _state['hubs'] if h.is_paralympic_hot_spot)}."
        )

    return f"Action {tool_name} executed with args {args}."

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        direct = _direct_chat_response(req)
        if direct:
            return direct

        client = genai.Client(
            vertexai=True,
            project="hometown-success-engine",
            location="global",
        )

        contents = []
        session_context = _session_context_text(req.session_id)
        if session_context:
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part(
                            text=(
                                "Recent session context for resolving follow-up references only. "
                                "Ground all facts in tools before answering:\n"
                                f"{session_context}"
                            )
                        )
                    ],
                )
            )
        for turn in req.history:
            role = turn.get("role", "user")
            text = turn.get("text", "")
            if text:
                contents.append(
                    genai_types.Content(
                        role=role,
                        parts=[genai_types.Part(text=text)],
                    )
                )
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=req.message)],
            )
        )

        # First pass: let Gemini decide whether to call a tool
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=_build_chatbot_system_prompt(),
                tools=[_build_chatbot_tools()],
                temperature=0.6,
                max_output_tokens=400,
            ),
        )

        text_parts: list[str] = []
        tool_calls: list[ChatToolCall] = []
        function_call_parts: list[genai_types.Part] = []

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(_prepare_tool_call_for_frontend(fc.name, dict(fc.args) if fc.args else {}))
                    function_call_parts.append(part)

        reply_text = " ".join(text_parts).strip()

        # If Gemini called tools, do a second turn so it can narrate the
        # tool results in natural language instead of dumping raw data.
        if tool_calls and function_call_parts:
            # Build the function response parts to feed back to Gemini
            function_response_parts = []
            for call, fc_part in zip(tool_calls, function_call_parts):
                rich_context = _build_tool_result_context(call.name, call.args)
                function_response_parts.append(
                    genai_types.Part.from_function_response(
                        name=call.name,
                        response={"result": rich_context},
                    )
                )

            # Append the model's tool-call turn and the user's tool-response turn
            second_turn_contents = list(contents)
            second_turn_contents.append(
                genai_types.Content(
                    role="model",
                    parts=function_call_parts,
                )
            )
            second_turn_contents.append(
                genai_types.Content(
                    role="user",
                    parts=function_response_parts,
                )
            )

            try:
                second_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=second_turn_contents,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_build_chatbot_system_prompt(),
                        tools=[_build_chatbot_tools()],
                        temperature=0.6,
                        max_output_tokens=400,
                    ),
                )
                second_text_parts = []
                if second_response.candidates and second_response.candidates[0].content:
                    for part in second_response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            second_text_parts.append(part.text)
                narrated = " ".join(second_text_parts).strip()
                if narrated:
                    reply_text = narrated
                elif not reply_text:
                    # Gemini didn't narrate. Fall back to clean tool result.
                    summaries = [_build_tool_result_context(c.name, c.args) for c in tool_calls]
                    reply_text = " ".join(summaries)
            except Exception as e:
                logger.warning(f"Second-turn narration failed: {e}")
                if not reply_text:
                    summaries = [_build_tool_result_context(c.name, c.args) for c in tool_calls]
                    reply_text = " ".join(summaries)

        if not reply_text:
            reply_text = (
                "I focused the map for you. Take a look ,  and ask me about "
                "any of the highlighted regions if you want to dig in."
            )

        new_history = list(req.history)
        new_history.append({"role": "user", "text": req.message})
        new_history.append({"role": "model", "text": reply_text})
        _remember_session_turn(req.session_id, req.message, reply_text, tool_calls)

        return ChatResponse(
            text=reply_text,
            tool_calls=tool_calls,
            history=new_history,
        )

    except Exception as e:
        import traceback
        logger.error(f"Chat endpoint error: {e}\n{traceback.format_exc()}")
        return ChatResponse(
            text=f"I ran into an issue: {str(e)[:120]}. Try rephrasing.",
            tool_calls=[],
            history=req.history,
        )


def _voice_speech_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if clean.startswith("Map zoomed/selected:"):
        hub_name = clean.split(" (", 1)[0].replace("Map zoomed/selected: ", "").strip()
        counts_match = re.search(
            r"(\d[\d,]* athletes total, \d[\d,]* Olympians, \d[\d,]* Paralympians, [\d.]+% Paralympic share\.)",
            clean,
        )
        top_sport_match = re.search(r"Top sport: ([^.]+)\.", clean)
        climate_match = re.search(r"Climate: avg (.*? elevation)\.", clean)
        parts = [f"{hub_name} is selected."]
        if counts_match:
            parts.append(counts_match.group(1))
        if top_sport_match:
            parts.append(f"Top sport: {top_sport_match.group(1)}.")
        if climate_match:
            climate_text = (
                climate_match.group(1)
                .replace("°F", " degrees Fahrenheit")
                .replace("in precip/yr", " inches of precipitation per year")
                .replace("ft elevation", " feet elevation")
            )
            parts.append(f"Climate context: {climate_text}.")
        parts.append("See the chat panel for rankings and full grounded detail.")
        return " ".join(parts)

    if len(clean) <= VOICE_TTS_MAX_CHARS:
        return clean

    clipped = clean[:VOICE_TTS_MAX_CHARS].rstrip()
    sentence_break = clipped.rfind(". ")
    if sentence_break > 320:
        clipped = clipped[:sentence_break + 1]
    return f"{clipped} See the chat panel for the full grounded detail."


def _synthesize_gemini_tts(text: str) -> tuple[str, str]:
    speech_text = _voice_speech_text(text)
    client = genai.Client(
        vertexai=True,
        project="hometown-success-engine",
        location=VOICE_LOCATION,
    )
    response = client.models.generate_content(
        model=VOICE_TTS_MODEL_ID,
        contents=(
            "Read this in a warm, natural analyst voice for a live product demo. "
            f"Keep the delivery clear and confident: {speech_text}"
        ),
        config=genai_types.GenerateContentConfig(
            response_modalities=[genai_types.Modality.AUDIO],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=VOICE_NAME
                    )
                )
            ),
        ),
    )

    if not response.candidates or not response.candidates[0].content:
        raise RuntimeError("Gemini TTS returned no audio candidate")
    for part in response.candidates[0].content.parts or []:
        if part.inline_data and part.inline_data.data:
            data = part.inline_data.data
            if isinstance(data, str):
                encoded = data
            else:
                encoded = base64.b64encode(data).decode("ascii")
            return encoded, part.inline_data.mime_type or "audio/L16;codec=pcm;rate=24000"
    raise RuntimeError("Gemini TTS returned no inline audio")


async def _send_voice_audio(
    websocket: WebSocket,
    audio_data: str,
    mime_type: str,
    source: str,
    turn_id: int | None = None,
) -> None:
    max_chunk_chars = 700_000
    if len(audio_data) <= max_chunk_chars:
        message: dict[str, Any] = {
            "type": "audio",
            "data": audio_data,
            "mime_type": mime_type,
            "source": source,
        }
        if turn_id:
            message["turn_id"] = turn_id
        await websocket.send_json(message)
        return

    audio_id = f"{source}-{abs(hash(audio_data))}"
    chunks = [
        audio_data[i:i + max_chunk_chars]
        for i in range(0, len(audio_data), max_chunk_chars)
    ]
    for index, chunk in enumerate(chunks):
        message = {
            "type": "audio_chunk",
            "id": audio_id,
            "index": index,
            "total": len(chunks),
            "data": chunk,
            "mime_type": mime_type,
            "source": source,
        }
        if turn_id:
            message["turn_id"] = turn_id
        await websocket.send_json(message)


@app.websocket("/voice/ws")
async def voice_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    client = genai.Client(
        vertexai=True,
        project="hometown-success-engine",
        location=VOICE_LOCATION,
    )
    voice_session_id = _session_id(str(websocket.query_params.get("session_id") or ""))
    voice_context = _session_context_text(voice_session_id)
    legacy_context = str(websocket.query_params.get("context") or "").strip()
    if legacy_context and not voice_context:
        voice_context = legacy_context[-1600:]
    voice_system_instruction = (
        f"{_build_chatbot_system_prompt()}\n\n"
        "VOICE MODE RULES:\n"
        "- You are speaking through Gemini Live native audio.\n"
        "- After every tool/function response, speak a concise grounded answer out loud.\n"
        "- Do not stop silently after a tool call.\n"
        "- Keep spoken answers short enough for a live demo, but include the key counts and map action."
    )
    if voice_context:
        voice_system_instruction = (
            f"{voice_system_instruction}\n\n"
            "RECENT VISIBLE CHAT CONTEXT FOR THIS VOICE TURN:\n"
            f"{voice_context}\n\n"
            "Use this only to resolve conversational references such as 'that hub', "
            "'there', or 'compare it'. Continue to ground all data in the runtime tools."
        )
    audio_enabled_for_turn = True
    voice_turn_id = 0
    voice_input_text = ""
    voice_output_text = ""
    voice_tool_calls: list[dict[str, Any]] = []
    voice_tool_result_texts: list[str] = []
    voice_audio_sent = False

    async def send_voice_state(
        state: str,
        label: str,
        detail: str = "",
    ) -> None:
        message: dict[str, Any] = {
            "type": "voice_state",
            "state": state,
            "label": label,
            "detail": detail,
        }
        if voice_turn_id:
            message["turn_id"] = voice_turn_id
        await websocket.send_json(message)

    async def send_live_messages(session) -> None:
        nonlocal voice_input_text, voice_output_text, voice_tool_calls, voice_tool_result_texts, voice_audio_sent
        async for message in session.receive():
            if message.setup_complete:
                await websocket.send_json({"type": "ready", "turn_id": voice_turn_id})
                await send_voice_state("idle", "Voice ready", "Gemini Live native audio is connected.")

            if message.server_content:
                content = message.server_content
                if content.interrupted:
                    await websocket.send_json({"type": "interrupted", "turn_id": voice_turn_id})
                    await send_voice_state("interrupted", "Interrupted", "Listening for your next question.")
                if content.input_transcription and content.input_transcription.text:
                    voice_input_text = content.input_transcription.text.strip() or voice_input_text
                    await websocket.send_json({
                        "type": "input_transcript",
                        "text": content.input_transcription.text,
                        "final": bool(getattr(content.input_transcription, "finished", False)),
                        "turn_id": voice_turn_id,
                    })
                if content.output_transcription and content.output_transcription.text:
                    voice_output_text = (
                        f"{voice_output_text} {content.output_transcription.text}".strip()
                        if voice_output_text else content.output_transcription.text.strip()
                    )
                    await websocket.send_json({
                        "type": "output_transcript",
                        "text": content.output_transcription.text,
                        "final": bool(getattr(content.output_transcription, "finished", False)),
                        "turn_id": voice_turn_id,
                    })
                    await send_voice_state("replying", "Gemini replying", content.output_transcription.text)
                if content.model_turn and content.model_turn.parts:
                    for part in content.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            if not audio_enabled_for_turn:
                                continue
                            data = part.inline_data.data
                            if isinstance(data, bytes):
                                data = base64.b64encode(data).decode("ascii")
                            voice_audio_sent = True
                            await websocket.send_json({
                                "type": "audio",
                                "data": data,
                                "mime_type": part.inline_data.mime_type or "audio/pcm;rate=24000",
                                "turn_id": voice_turn_id,
                            })
                if content.turn_complete:
                    remembered_text = voice_output_text or " ".join(voice_tool_result_texts)
                    fallback_text = (" ".join(voice_tool_result_texts) or voice_output_text).strip()
                    _remember_session_turn(
                        voice_session_id,
                        voice_input_text,
                        remembered_text,
                        voice_tool_calls,
                    )
                    if audio_enabled_for_turn and fallback_text and not voice_audio_sent:
                        try:
                            tts_audio, tts_mime_type = await asyncio.to_thread(
                                _synthesize_gemini_tts,
                                fallback_text,
                            )
                            await _send_voice_audio(
                                websocket,
                                tts_audio,
                                tts_mime_type,
                                "gemini-tts-fallback",
                                turn_id=voice_turn_id,
                            )
                            voice_audio_sent = True
                            await send_voice_state(
                                "replying",
                                "Gemini replying",
                                "Using Gemini audio fallback because native Live audio was silent.",
                            )
                        except Exception as tts_error:
                            logger.warning(f"Gemini voice fallback failed: {tts_error}")
                            await websocket.send_json({
                                "type": "speech_fallback",
                                "text": fallback_text,
                                "turn_id": voice_turn_id,
                            })
                            await send_voice_state(
                                "replying",
                                "Gemini replying",
                                "Using browser audio fallback because Gemini audio fallback failed.",
                            )
                    await websocket.send_json({
                        "type": "turn_complete",
                        "turn_id": voice_turn_id,
                    })
                    await send_voice_state("idle", "Voice ready", "Ask another question or press the mic.")

            if message.tool_call and message.tool_call.function_calls:
                frontend_calls: list[dict[str, Any]] = []
                function_responses = []
                tool_result_texts: list[str] = []
                for call in message.tool_call.function_calls:
                    args = dict(call.args) if call.args else {}
                    tool_name = call.name or ""
                    frontend_call = _prepare_tool_call_for_frontend(tool_name, args)
                    frontend_calls.append(frontend_call.model_dump())
                    result = _build_tool_result_context(frontend_call.name, frontend_call.args)
                    tool_result_texts.append(result)
                    function_responses.append(
                        genai_types.FunctionResponse(
                            id=call.id,
                            name=tool_name,
                            response={"result": result},
                        )
                    )
                voice_tool_calls = frontend_calls
                voice_tool_result_texts = tool_result_texts
                await websocket.send_json({
                    "type": "tool_calls",
                    "tool_calls": frontend_calls,
                    "turn_id": voice_turn_id,
                })
                if tool_result_texts:
                    tool_result_text = " ".join(tool_result_texts)
                    await send_voice_state(
                        "tool",
                        "Moving map",
                        "Gemini called the map/data tools and is preparing a grounded spoken answer.",
                    )
                    await websocket.send_json({
                        "type": "tool_result_text",
                        "text": tool_result_text,
                        "speak_fallback": False,
                        "turn_id": voice_turn_id,
                    })
                if function_responses:
                    await session.send_tool_response(function_responses=function_responses)
                    await send_voice_state("replying", "Gemini replying", "Grounded map result received.")

    try:
        await send_voice_state("connecting", "Connecting to Gemini Live", "")
        async with client.aio.live.connect(
            model=VOICE_MODEL_ID,
            config=genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.AUDIO],
                system_instruction=voice_system_instruction,
                tools=[_build_chatbot_tools()],
                temperature=0.4,
                realtime_input_config=genai_types.RealtimeInputConfig(
                    automatic_activity_detection=genai_types.AutomaticActivityDetection(
                        disabled=False,
                        start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_LOW,
                        end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_LOW,
                        prefix_padding_ms=20,
                        silence_duration_ms=600,
                    ),
                    activity_handling=genai_types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                ),
                context_window_compression=genai_types.ContextWindowCompressionConfig(
                    sliding_window=genai_types.SlidingWindow(),
                ),
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name=VOICE_NAME
                        )
                    )
                ),
                output_audio_transcription=genai_types.AudioTranscriptionConfig(),
                input_audio_transcription=genai_types.AudioTranscriptionConfig(),
            ),
        ) as session:
            await websocket.send_json({
                "type": "connecting",
                "model": VOICE_MODEL_ID,
                "turn_id": voice_turn_id,
            })
            await websocket.send_json({"type": "ready", "turn_id": voice_turn_id})
            await send_voice_state("idle", "Gemini Live voice ready", "Press the mic to ask a spoken question.")
            receiver = asyncio.create_task(send_live_messages(session))
            try:
                while True:
                    payload = await websocket.receive_json()
                    message_type = payload.get("type")
                    if message_type == "text":
                        if "audio_enabled" in payload:
                            audio_enabled_for_turn = bool(payload.get("audio_enabled"))
                        text = str(payload.get("text") or "").strip()
                        if text:
                            incoming_turn = payload.get("turn_id")
                            if isinstance(incoming_turn, int) and incoming_turn > 0:
                                voice_turn_id = max(voice_turn_id, incoming_turn)
                            else:
                                voice_turn_id += 1
                            voice_input_text = text
                            voice_output_text = ""
                            voice_tool_calls = []
                            voice_tool_result_texts = []
                            voice_audio_sent = False
                            await websocket.send_json({
                                "type": "turn_started",
                                "turn_id": voice_turn_id,
                                "input": "text",
                            })
                            await send_voice_state("thinking", "Gemini Live is answering", text)
                            await websocket.send_json({
                                "type": "input_transcript",
                                "text": text,
                                "final": True,
                                "turn_id": voice_turn_id,
                            })
                            await session.send_client_content(
                                turns=genai_types.Content(
                                    role="user",
                                    parts=[genai_types.Part(text=text)],
                                ),
                                turn_complete=True,
                            )
                    elif message_type == "audio_start":
                        if "audio_enabled" in payload:
                            audio_enabled_for_turn = bool(payload.get("audio_enabled"))
                        incoming_turn = payload.get("turn_id")
                        if isinstance(incoming_turn, int) and incoming_turn > 0:
                            voice_turn_id = max(voice_turn_id, incoming_turn)
                        else:
                            voice_turn_id += 1
                        voice_input_text = ""
                        voice_output_text = ""
                        voice_tool_calls = []
                        voice_tool_result_texts = []
                        voice_audio_sent = False
                        await websocket.send_json({
                            "type": "turn_started",
                            "turn_id": voice_turn_id,
                            "input": "audio",
                        })
                        await send_voice_state("listening", "Listening", "Speak naturally. Gemini Live is streaming audio.")
                    elif message_type == "audio_chunk":
                        if "audio_enabled" in payload:
                            audio_enabled_for_turn = bool(payload.get("audio_enabled"))
                        incoming_turn = payload.get("turn_id")
                        if isinstance(incoming_turn, int) and incoming_turn > 0:
                            if incoming_turn < voice_turn_id:
                                continue
                            voice_turn_id = incoming_turn
                        data = str(payload.get("data") or "")
                        if data:
                            mime_type = str(payload.get("mime_type") or "audio/pcm;rate=16000")
                            try:
                                raw_audio = base64.b64decode(data)
                            except Exception:
                                await send_voice_state("error", "Audio chunk skipped", "Invalid microphone audio payload.")
                                continue
                            await session.send_realtime_input(
                                audio=genai_types.Blob(
                                    data=raw_audio,
                                    mime_type=mime_type,
                                )
                            )
                    elif message_type == "audio_end":
                        if "audio_enabled" in payload:
                            audio_enabled_for_turn = bool(payload.get("audio_enabled"))
                        incoming_turn = payload.get("turn_id")
                        if isinstance(incoming_turn, int) and incoming_turn > 0:
                            if incoming_turn < voice_turn_id:
                                continue
                            voice_turn_id = incoming_turn
                        await session.send_realtime_input(audio_stream_end=True)
                        await send_voice_state("thinking", "Gemini Live is thinking", "Audio turn ended.")
                    elif message_type == "close":
                        break
            except WebSocketDisconnect:
                pass
            finally:
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver
                await session.close()
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)[:200],
            })
        except Exception:
            pass

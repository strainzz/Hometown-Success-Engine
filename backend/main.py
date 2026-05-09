import asyncio
import json
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from google import genai
from google.genai import types as genai_types

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# v2 build

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

def state_from_latlon(lat: float, lon: float) -> str:
    """Returns 2-letter US state/territory code for a given lat/lon.
    Returns 'XX' as a last-resort fallback only when coordinates are
    outside all known US bounding boxes."""
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


class ChatToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal[
        "select_hub",
        "filter_to_paralympic",
        "zoom_to_hub",
        "reset_view",
        "select_state",
        "query_data",
    ]
    args: dict


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)


CHATBOT_TOOLS = genai_types.Tool(
    function_declarations=[
        genai_types.FunctionDeclaration(
            name="query_data",
            description="Look up specific data without moving the map. Use for ranking, comparison, top-N, or aggregate questions like 'which state ranks first', 'top 5 states for Paralympic representation', 'how many hubs are in the Pacific region', 'what are the most common sports in the Mountain West'. Use this whenever the user asks a data question that does NOT require navigating the map.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={
                    "query_type": genai_types.Schema(
                        type="STRING",
                        enum=[
                            "top_states_by_total",
                            "top_states_by_paralympic",
                            "top_states_by_paralympic_share",
                            "top_hubs_by_total",
                            "top_hubs_by_paralympic_share",
                            "all_hot_spots",
                            "hubs_by_sport",
                            "hubs_by_macro_region",
                            "summary",
                        ],
                        description="What kind of lookup. top_states_by_total/paralympic/paralympic_share return ranked state lists. top_hubs_by_total/paralympic_share return ranked hub lists. all_hot_spots returns the 5 Paralympic Hot Spots. hubs_by_sport filters by a sport name. hubs_by_macro_region filters by region. summary returns overall totals.",
                    ),
                    "limit": genai_types.Schema(
                        type="INTEGER",
                        description="How many results to return (default 5, max 20).",
                    ),
                    "sport": genai_types.Schema(
                        type="STRING",
                        description="Sport name for hubs_by_sport queries (e.g. 'swimming', 'cross-country skiing', 'wheelchair basketball').",
                    ),
                    "macro_region": genai_types.Schema(
                        type="STRING",
                        description="Macro region for hubs_by_macro_region queries (e.g. 'Pacific', 'Mountain West', 'Midwest').",
                    ),
                },
                required=["query_type"],
            ),
        ),
        genai_types.FunctionDeclaration(
            name="select_state",
            description="Open the state info panel for a US state, territory, or DC. Use when the user asks about a specific state by name or code (e.g. 'show me Kansas', 'go to D.C.', 'tell me about Wyoming', 'Texas?'). The panel shows total athletes, Paralympic count, rank, and top hub in that state.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={
                    "state_code": genai_types.Schema(
                        type="STRING",
                        description="2-letter US state postal code. Examples: CA, TX, KS, DC (District of Columbia), PR (Puerto Rico), AK, HI. Always uppercase.",
                    ),
                },
                required=["state_code"],
            ),
        ),
    ]
)

CHATBOT_SYSTEM_PROMPT = """You are Gemini, the AI guide for the Hometown Success Engine ,  an interactive map of where America's elite athletes come from. The map shows 5,012 Olympians and Paralympians clustered into 37 regional hubs.

# YOUR JOB
Help users explore the map. Call tools to drive it. Then narrate what changed using SPECIFIC NUMBERS from the tool results.

# AVAILABLE TOOLS
- select_hub(hub_id) ,  highlights a hub
- zoom_to_hub(hub_id) ,  zooms to a hub
- filter_to_paralympic(macro_region?) ,  highlights Paralympic Hot Spots
- select_state(state_code) ,  opens the state panel for any US state, DC, or territory
- reset_view() ,  resets to continental US

# WHEN TO USE STATE vs HUB vs QUERY
- If user names a STATE (Kansas, Wyoming, D.C., Texas, California): use select_state
- If user names a CITY or REGION (Phoenix, Anchorage, Stillwater, Lincoln, Merced): use select_hub
- If user asks for RANKINGS, TOP-N, COMPARISONS, or AGGREGATE questions ("which state ranks first", "top 5 states", "most Paralympic athletes", "how many hubs", "what are the Hot Spots"): use query_data
- States have aggregated athlete counts; hubs are real geographic clusters with narratives; query_data is for analyst-style lookups that don't fit on the map directly.

# MULTI-TOOL CHAINING (MANDATORY)

Some user messages REQUIRE you to call MULTIPLE tools in a single response. The function-calling API supports parallel/sequential tool calls in one turn — use this capability.

TRIGGER PHRASES that REQUIRE 2 tool calls:
- "zoom in on the top/best/highest/leading [state/hub/region]"
- "show me the [#1/leader/top] for [metric]"
- "go to the state with the most [X]"
- "take me to the highest-ranked [state/hub]"

When you see these patterns, you MUST emit BOTH function calls in the same turn:
1. query_data to identify the answer (e.g. top_states_by_paralympic_share, limit=1)
2. select_state OR select_hub with the result of step 1 (e.g. state_code="NV")

If the user wants STATE-level data: pair query_data with select_state.
If the user wants HUB-level data: pair query_data with select_hub.

DO NOT respond with only query_data when the user asked you to navigate. Both tools must fire.

# THE 5 PARALYMPIC HOT SPOTS
- HUB_AZ_PHOENIX (12.7%), HUB_AK_ANCHORAGE (12.5%), HUB_NE_LINCOLN (10.8%), HUB_OK_STILLWATER (9.3%), HUB_CA_MERCED (9.3%)
- National Paralympic baseline: 4.6%
- Hot Spots = >2x national rate

# RESPONSE RULES ,  STRICT
1. ALWAYS call a tool when the user mentions a place, region, or filter.
2. After tool calls, your text MUST quote SPECIFIC numbers from the tool result. If the tool result says "12.7%", you say "12.7%". If it says "55 athletes", you say "55 athletes". Generic phrases like "significantly higher" without a number are FORBIDDEN.
3. Every response must be 2-3 sentences. Never one sentence. Never one word.
4. When the tool result includes climate or geographic_context, weave ONE specific climate fact (e.g. "74°F average", "10 inches of rain") into your narration to ground the answer in real geography. Never claim geography "produces" athletes; use conditional phrasing like "could help foster" or "may explain".
5. Always end with a suggestion for what to explore next ("Click any red hub..." / "Want me to zoom to Phoenix?" / "Try the Pacific region next.")
6. Off-topic questions (weather, sports scores, news): redirect to map exploration.

# RESPONSE FORMAT (FOLLOW EXACTLY)
[Sentence 1: what changed on the map, with specific numbers from tool result]
[Sentence 2: one interesting fact, with specific numbers]
[Sentence 3: suggested next action]

# EXAMPLES

User: "Show Paralympic Hot Spots"
Tool: filter_to_paralympic()
Tool result: "Filter applied: highlighting all 5 Paralympic Hot Spots. Top spot: Phoenix Region at 12.7% Paralympic (55 athletes). All 5: Phoenix Region (12.7%), Anchorage Region (12.5%), Lincoln Region (10.8%), Stillwater Region (9.3%), Merced Region (9.3%)."
Your response: "Highlighted the 5 Paralympic Hot Spots ,  regions with more than 2x the national 4.6% Paralympic rate. Phoenix leads at 12.7% with 55 athletes, followed by Anchorage at 12.5%. Click any red hub to read its full story."

User: "Tell me about Anchorage"
Tool: select_hub(hub_id="HUB_AK_ANCHORAGE")
Tool result: "Map zoomed/selected: Anchorage Region (Anchorage, AK, Pacific). 24 athletes total, 12.5% Paralympic. Top sport: Skiing. This IS a Paralympic Hot Spot. Climate: avg 36.7°F, 39.8in precip/yr, 102ft elevation. Geographic context: The consistently cold annual average temperature and abundant snowfall in Southcentral Alaska could help foster a strong culture around winter sports."
Your response: "Zoomed to Anchorage, one of 5 Paralympic Hot Spots at 12.5% Paralympic with 24 athletes. The cold average temperature of 36.7°F and abundant snowfall could help foster the winter sports culture that anchors this hub, especially skiing and ice hockey. Want to see how Phoenix compares as a Hot Spot in a totally different climate?"

User: "What's the weather?"
Your response: "I focus on this map of America's elite athletes. Want me to show you a region or zoom to a Paralympic Hot Spot instead?"

User: "Tell me about Kansas"
Tool: select_state(state_code="KS")
Tool result: "State panel opened: Kansas (KS). 59 athletes total, 6 Paralympians, 10.2% Paralympic share. This is above the 4.6% national baseline. Total athletes rank: #34. Paralympic rank: #29. Top hub in Kansas: Stillwater Region with 86 athletes (9.3% Paralympic, a Paralympic Hot Spot)."
Your response: "Kansas has 59 Olympians and Paralympians in our dataset, with a 10.2% Paralympic share that runs more than 2x the national baseline. The state is anchored by the Stillwater Region, one of our 5 Paralympic Hot Spots at 9.3% Paralympic. Want to zoom into Stillwater itself?"

User: "Which state has the highest Paralympic representation?"
Tool: query_data(query_type="top_states_by_paralympic_share", limit=3)
Tool result: "Top 3 states by Paralympic share (states with 10+ athletes only, national baseline 4.6%): 1. Kansas (KS) - 10.2% Paralympic (6 of 59 athletes) 2. Maryland (MD) - 8.5% Paralympic (X of Y) 3. Oklahoma (OK) - 7.1% Paralympic (X of Y)"
Your response: "Kansas leads our 52 mappable regions with 10.2% Paralympic share, more than 2x the 4.6% national baseline. Maryland follows at 8.5% and Oklahoma at 7.1%. Want me to open the Kansas state panel or filter the map to the 5 Paralympic Hot Spots?"

User: "What are the top 5 hubs by Paralympic share?"
Tool: query_data(query_type="top_hubs_by_paralympic_share", limit=5)
Your response: [narrate the 5 results with specific percentages and end with a follow-up suggestion]

# TONE
Warm, knowledgeable, specific. Never generic. Always cite numbers from the tool result.

# AFTER CALLING A TOOL: ALWAYS NARRATE
After ANY tool returns its result, you MUST produce a 1-2 sentence natural-language reply that explains the result conversationally. Never reply with empty text after a tool call.

Examples:
- After query_data returns "Top 1 states by Paralympic share: 1. Nevada (NV) - 13.5% Paralympic (7 of 52 athletes)":
  Reply: "Nevada leads with 13.5% Paralympic representation, more than 2.9x the 4.6% national baseline. Want me to zoom to Nevada or compare it to the next-ranked states?"
- After select_hub returns hub data:
  Reply: A natural narrative using the climate, sports, and Paralympic data from the result.
- After all_hot_spots returns the 5 Hot Spots:
  Reply: Summarize the geographic spread (Phoenix in the Southwest, Anchorage in Alaska, Lincoln in the Plains, etc.) and end with a follow-up question.

NEVER respond with just "I focused the map for you" — always provide specific numbers and insight from the tool result.
"""

_state: dict[str, Any] = {
    "hubs": [],
    "hubs_by_id": {},
    "narratives": {},
    "athletes_geo_points": [],
    "state_aggregates": [],
}


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
        c_hubs = root / "pipeline" / "clustered" / "hubs.json"
        c_narratives = root / "pipeline" / "narratives" / "hubs.json"
        c_athletes = root / "pipeline" / "clustered" / "athletes.json"
        if c_hubs.exists() and c_narratives.exists() and c_athletes.exists():
            hubs_path = c_hubs
            narratives_path = c_narratives
            athletes_path = c_athletes
            logger.info(f"Loading data from: {root}")
            break

    if hubs_path is None:
        tried = [str(r) for r in candidate_roots]
        raise RuntimeError(
            f"Data files not found in any candidate location: {tried}"
        )

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

    logger.info(
        f"Loaded {len(hubs)} hubs, {len(narratives)} narratives, {len(athletes_geo_points)} athletes. "
        f"Aggregated {len(state_aggregates)} states. "
        f"{sum(1 for h in hubs if h.is_paralympic_hot_spot)} Paralympic Hot Spots."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_data()
    yield


app = FastAPI(
    title="Hometown Success Engine API",
    description="Team USA hometown hubs with regional context and chatbot tools",
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

    if tool_name == "filter_to_paralympic":
        macro_region = args.get("macro_region")
        hot_spots = [h for h in _state["hubs"] if h.is_paralympic_hot_spot]
        if macro_region:
            filtered = [h for h in hot_spots if h.macro_region == macro_region]
            if not filtered:
                return (
                    f"The Paralympic filter was applied for the {macro_region} region, "
                    f"but no Paralympic Hot Spots are present there. The 5 national Hot Spots "
                    f"are in: " + ", ".join(f"{h.display_name}" for h in hot_spots) + "."
                )
            top = max(filtered, key=lambda h: h.composition.paralympic_share)
            return (
                f"Filter applied: highlighting Paralympic Hot Spots in {macro_region}. "
                f"Hubs visible: {', '.join(h.display_name for h in filtered)}. "
                f"Leading: {top.display_name} at {top.composition.paralympic_share*100:.1f}% Paralympic, "
                f"{top.total_athletes} athletes total. National Paralympic baseline: 4.6%."
            )
        else:
            sorted_hot = sorted(hot_spots, key=lambda h: -h.composition.paralympic_share)
            top = sorted_hot[0]
            return (
                f"Filter applied: highlighting all 5 Paralympic Hot Spots ,  regions where "
                f"Paralympic athletes are more than 2x the 4.6% national baseline. "
                f"Top spot: {top.display_name} at {top.composition.paralympic_share*100:.1f}% Paralympic "
                f"({top.total_athletes} athletes). All 5: " +
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
        top_sport = hub.top_sports[0].sport if hub.top_sports else "various sports"
        result = (
            f"Map zoomed/selected: {hub.display_name} ({hub.region_name}, {hub.macro_region}). "
            f"{hub.total_athletes} athletes total, {para_pct:.1f}% Paralympic. "
            f"Top sport: {top_sport}. "
            f"{'This IS a Paralympic Hot Spot.' if hub.is_paralympic_hot_spot else 'Not a Paralympic Hot Spot.'}"
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
                    f"The 5 Paralympic Hot Spots are all elsewhere."
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
            above_or_below = "above" if para_pct > 4.6 else "below"
            parts = [
                f"State panel opened: {name} ({code}).",
                f"{agg.total_athletes} athletes total, {para} Paralympians, {para_pct:.1f}% Paralympic share.",
                f"This is {above_or_below} the 4.6% national baseline.",
                f"Total athletes rank: #{total_rank}. Paralympic rank: #{para_rank}.",
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

        aggregates = _state["state_aggregates"]
        hubs = _state["hubs"]

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
            lines = [f"Top {limit} states by Paralympic share (states with 25+ athletes for statistical reliability, national baseline 4.6%):"]
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
            lines = [f"Top {limit} hubs by Paralympic share (national baseline 4.6%):"]
            for i, h in enumerate(sorted_hubs, 1):
                hot = " (Hot Spot)" if h.is_paralympic_hot_spot else ""
                lines.append(f"{i}. {h.display_name}{hot} - {h.composition.paralympic_share*100:.1f}% Paralympic, {h.total_athletes} athletes")
            return " ".join(lines)

        if query_type == "all_hot_spots":
            hot_spots = sorted([h for h in hubs if h.is_paralympic_hot_spot], key=lambda h: -h.composition.paralympic_share)
            lines = [f"All {len(hot_spots)} Paralympic Hot Spots (regions where Paralympic share runs 2x+ the 4.6% national baseline):"]
            for i, h in enumerate(hot_spots, 1):
                top_sport = h.top_sports[0].sport if h.top_sports else "various sports"
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
                return f"No hubs found with '{sport}' as a top sport across our 37 hubs."
            lines = [f"Top {len(matches)} hubs where '{sport}' appears among top sports:"]
            for i, (h, sp) in enumerate(matches, 1):
                lines.append(f"{i}. {h.display_name} - {sp.count} {sp.sport} athletes ({sp.paralympic_count} Paralympic), hub total {h.total_athletes}")
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
                f"Dataset summary: {total_athletes} Olympians and Paralympians from the 2020 to 2026 cycle, "
                f"clustered into {len(hubs)} hometown hubs across {states_count} US states and territories. "
                f"{hot_spots} regions qualify as Paralympic Hot Spots (Paralympic share 2x+ above the 4.6% national baseline). "
                f"Overall Paralympic share across the dataset: {overall_para_pct:.1f}%."
            )

        return f"Unknown query_type: {query_type}"

    if tool_name == "reset_view":
        return (
            "Map reset to default continental US view. All filters cleared. "
            f"Total athletes mapped: {sum(h.total_athletes for h in _state['hubs'])}. "
            f"Total hubs: {len(_state['hubs'])}. "
            f"Total Paralympic Hot Spots: {sum(1 for h in _state['hubs'] if h.is_paralympic_hot_spot)}."
        )

    return f"Action {tool_name} executed with args {args}."

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        client = genai.Client(
            vertexai=True,
            project="hometown-success-engine",
            location="global",
        )

        contents = []
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
                system_instruction=CHATBOT_SYSTEM_PROMPT,
                tools=[CHATBOT_TOOLS],
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
                    tool_calls.append(
                        ChatToolCall(
                            name=fc.name,
                            args=dict(fc.args) if fc.args else {},
                        )
                    )
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
                        system_instruction=CHATBOT_SYSTEM_PROMPT,
                        tools=[CHATBOT_TOOLS],
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

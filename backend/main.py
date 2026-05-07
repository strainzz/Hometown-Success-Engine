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
    (35.00, 40.00, -103.00, -94.43, "OK"), # OK before TX/KS
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
    (41.00, 49.00, -116.05, -104.04, "MT"),
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


class HubNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hub_id: str
    display_name: str
    headline: str
    summary: str
    paralympic_callout: Optional[str] = None
    top_sport_phrase: str
    confidence_qualifier: str


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
            name="select_hub",
            description="Select a specific hometown hub on the map by its hub_id. Use when user asks about a specific city, region, or named hub.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={
                    "hub_id": genai_types.Schema(
                        type="STRING",
                        description="The hub identifier, e.g. HUB_AZ_PHOENIX, HUB_CA_MERCED, HUB_AK_ANCHORAGE, HUB_NE_LINCOLN, HUB_OK_STILLWATER",
                    ),
                },
                required=["hub_id"],
            ),
        ),
        genai_types.FunctionDeclaration(
            name="filter_to_paralympic",
            description="Filter the map to highlight Paralympic Hot Spots, optionally narrowed to a specific macro_region. Use when user asks about Paralympic athletes, Paralympic representation, or wants to see Para Hot Spots.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={
                    "macro_region": genai_types.Schema(
                        type="STRING",
                        enum=["Pacific", "Mountain", "Plains", "Midwest", "Northeast", "South", "Caribbean"],
                        description="Optional macro region to filter to. Pacific includes AK, HI, WA, OR, CA. Mountain includes NV, AZ, NM, UT, CO, MT, ID, WY.",
                    ),
                },
            ),
        ),
        genai_types.FunctionDeclaration(
            name="zoom_to_hub",
            description="Zoom the map to a specific hub at higher zoom level. Use when user wants to focus on or see closer detail of a single hub.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={
                    "hub_id": genai_types.Schema(
                        type="STRING",
                        description="The hub identifier to zoom to.",
                    ),
                },
                required=["hub_id"],
            ),
        ),
        genai_types.FunctionDeclaration(
            name="reset_view",
            description="Reset the map to the default continental US view, clearing any selections or filters. Use when user asks to start over, see everything, or reset.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={},
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
- reset_view() ,  resets to continental US

# THE 5 PARALYMPIC HOT SPOTS
- HUB_AZ_PHOENIX (12.7%), HUB_AK_ANCHORAGE (12.5%), HUB_NE_LINCOLN (10.8%), HUB_OK_STILLWATER (9.3%), HUB_CA_MERCED (9.3%)
- National Paralympic baseline: 4.6%
- Hot Spots = >2x national rate

# RESPONSE RULES ,  STRICT
1. ALWAYS call a tool when the user mentions a place, region, or filter.
2. After tool calls, your text MUST quote SPECIFIC numbers from the tool result. If the tool result says "12.7%", you say "12.7%". If it says "55 athletes", you say "55 athletes". Generic phrases like "significantly higher" without a number are FORBIDDEN.
3. Every response must be 2-3 sentences. Never one sentence. Never one word.
4. Always end with a suggestion for what to explore next ("Click any red hub..." / "Want me to zoom to Phoenix?" / "Try the Pacific region next.")
5. Off-topic questions (weather, sports scores, news): redirect to map exploration.

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
Tool result: "Map zoomed/selected: Anchorage Region (Anchorage, AK, Pacific). 24 athletes total, 12.5% Paralympic. Top sport: Skiing. This IS a Paralympic Hot Spot."
Your response: "Zoomed to the Anchorage Region ,  Alaska's only hub on the map and one of just 5 national Paralympic Hot Spots. With 24 athletes and 12.5% representing Team USA at the Paralympic level, this is one of the most Paralympic-dense regions in the country, anchored by skiing. The narrative card below has the full story."

User: "What's the weather?"
Your response: "I focus on this map of America's elite athletes. Want me to show you a region or zoom to a Paralympic Hot Spot instead?"

# TONE
Warm, knowledgeable, specific. Never generic. Always cite numbers from the tool result.
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
        return result

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

        # If Gemini called tools but returned no text, use the rich tool result
        # context directly as the reply. No second Vertex AI call needed.
        if tool_calls and not reply_text:
            tool_summaries = []
            for call in tool_calls:
                rich_context = _build_tool_result_context(call.name, call.args)
                tool_summaries.append(rich_context)
            reply_text = " ".join(tool_summaries)

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

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HubNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hub_id: str
    display_name: str
    headline: str = Field(min_length=10, max_length=80)
    summary: str = Field(min_length=200, max_length=900)
    paralympic_callout: str | None = None
    top_sport_phrase: str = Field(min_length=10, max_length=120)
    confidence_qualifier: Literal[
        "could help find",
        "may foster",
        "is associated with",
        "appears to support",
    ]


FORBIDDEN_TERMS = [
    r"\bguarantees?\b",
    r"\bwheelchair[- ]bound\b",
    r"\bdisabled athletes?\b",
    r"\bsuffers? from\b",
    r"\bafflicted\b",
    r"\bdifferently[- ]abled\b",
    r"\bhandicapped\b",
    r"\bproduces\b(?!.*\bcould\b)",
    r"\bcreates winners\b",
]

REQUIRED_QUALIFIERS = {
    "could help find",
    "may foster",
    "is associated with",
    "appears to support",
}

SYSTEM_INSTRUCTION = """You are the Hometown Success Engine narrator for a public
Team USA hometown hub map. You generate hub profiles for an interactive
map showing where mapped Team USA Olympians and Paralympians are from.

NON-NEGOTIABLE RULES:
1. Use ONLY conditional phrasing: 'could help find', 'may foster',
  'is associated with', 'appears to support'. NEVER 'produces',
  'guarantees', 'creates winners', 'leads to'.
2. NEVER name individual athletes. Use aggregate references only.
3. IPC inclusive language: 'Paralympians', 'Paralympic athletes',
  'Paralympians'. FORBIDDEN: 'disabled athlete', 'wheelchair-bound',
  'suffers from', 'afflicted with', 'differently-abled', 'handicapped'.
4. If paralympic_share > 0, mention Paralympians in the FIRST TWO
  sentences of the summary. This is the equal-prominence rule.
5. Summary must be 80-120 words.
6. REGIONAL CONTEXT: The hub_data includes region_name (the
  locally recognizable regional name like "Valley of the Sun",
  "Mid-South / Mississippi Delta", or "Southcentral Alaska") and
  macro_region (one of: Northeast, Mid-Atlantic, South, Midwest,
  Southwest, Mountain West, Pacific, Alaska, Hawaii, Territories).
  USE the region_name in the headline OR first sentence of summary
  to ground the profile in recognizable geography. Examples:
  - "Phoenix, anchored in the Valley of the Sun, is associated with..."
  - "The Bay Area, connected to Northern California, may foster..."
  - "Stillwater sits in Oklahoma's Frontier Country, where..."
7. Ground all claims to the provided hub_data JSON. No invented stats.
8. If is_paralympic_hot_spot is true, the paralympic_callout field is
  REQUIRED. Hot Spots are hubs with 7.5% or higher Paralympic share;
  explain the regional Paralympic strength
  without comparing to Olympic counts.
9. headline must be specific, 10-80 chars, no clickbait.
10. top_sport_phrase format: 'could help find [SPORT1], [SPORT2], and
  [SPORT3]' - use the top 3 sports from the hub data.
11. confidence_qualifier must be one of:
  'could help find', 'may foster', 'is associated with',
  'appears to support'.
12. Avoid hype language such as elite, excellence, world-class,
  guaranteed, stars, champions, or next generation. Prefer clear
  product language: mapped athletes, competitors, hometown hubs,
  regional patterns, training environments, and public roster facts.

Generate one HubNarrative JSON object that passes all rules."""

MODEL_ID = "gemini-3.1-pro-preview"
FALLBACK_MODEL_ID = "gemini-2.5-pro"


class ComplianceError(Exception):
    def __init__(self, hub_id: str, violations: list[str]):
        self.hub_id = hub_id
        self.violations = violations
        super().__init__(f"Compliance violations in {hub_id}: {violations}")


def check_compliance(narrative: HubNarrative) -> list[str]:
    """Returns list of violation strings. Empty list = compliant."""
    violations = []
    components = [
        narrative.headline,
        narrative.summary,
        narrative.paralympic_callout or "",
        narrative.top_sport_phrase
    ]
    combined_text = " ".join(components).lower()

    for pattern in FORBIDDEN_TERMS:
        if re.search(pattern, combined_text):
            violations.append(f"Forbidden term matched: {pattern}")

    if narrative.confidence_qualifier not in REQUIRED_QUALIFIERS:
        violations.append(f"Invalid confidence qualifier: {narrative.confidence_qualifier}")

    return violations


async def generate_narrative(hub: dict, client: genai.Client) -> tuple[HubNarrative, int]:
    """Generate one narrative for one hub. Raises on compliance failure."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=HubNarrative,
        temperature=0.4,
        max_output_tokens=8192,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_ID,
            contents=json.dumps(hub),
            config=config,
        )
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e):
            logger.warning(
                f"Primary model {MODEL_ID} unavailable for "
                f"{hub.get('hub_id')}, falling back to {FALLBACK_MODEL_ID}"
            )
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=FALLBACK_MODEL_ID,
                contents=json.dumps(hub),
                config=config,
            )
        else:
            raise

    if not response.text:
        raise ValueError(f"Empty response from Gemini for {hub.get('hub_id')}")

    finish_reason = None
    if response.candidates and response.candidates[0].finish_reason:
        finish_reason = str(response.candidates[0].finish_reason)
    
    if finish_reason and "MAX_TOKENS" in finish_reason:
        logger.error(
            f"Response truncated for {hub.get('hub_id')} - hit max_output_tokens. "
            f"Increase max_output_tokens config."
        )
        raise ValueError(f"Truncated response for {hub.get('hub_id')}")

    try:
        narrative = HubNarrative.model_validate_json(response.text)
    except Exception as e:
        logger.error(
            f"JSON validation failed for {hub.get('hub_id')}: {e}. "
            f"Raw response (first 300 chars): {response.text[:300]}"
        )
        raise
    
    violations = check_compliance(narrative)
    if violations:
        raise ComplianceError(hub.get("hub_id", "UNKNOWN"), violations)

    token_count = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        token_count = getattr(response.usage_metadata, "total_token_count", 0)

    return narrative, token_count


async def process_hub_with_semaphore(
    hub: dict,
    client: genai.Client,
    semaphore: asyncio.Semaphore
) -> tuple[HubNarrative, int]:
    async with semaphore:
        hub_id = hub.get("hub_id", "UNKNOWN")
        athletes_count = hub.get("total_athletes", 0)
        para_share = hub.get("composition", {}).get("paralympic_share", 0.0)
        
        logger.info(f"Generating narrative for {hub_id} ({athletes_count} athletes, {para_share:.1%} para)")
        return await generate_narrative(hub, client)


async def main() -> None:
    base_dir = Path("pipeline")
    in_path = base_dir / "clustered" / "hubs.json"
    out_path = base_dir / "narratives" / "hubs.json"

    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        return

    with in_path.open("r", encoding="utf-8") as f:
        hubs = json.load(f)

    hubs.sort(key=lambda x: x.get("total_athletes", 0), reverse=True)

    client = genai.Client(
        vertexai=True,
        project=(
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT_ID")
            or "hometown-success-engine"
        ),
        location=os.environ.get("GEMINI_LOCATION", "global"),
    )
    
    semaphore = asyncio.Semaphore(4)
    tasks = [process_hub_with_semaphore(hub, client, semaphore) for hub in hubs]

    logger.info(f"Starting narrative generation for {len(hubs)} hubs...")
    
    try:
        results = await asyncio.gather(*tasks)
    except ComplianceError as e:
        logger.error(f"Generation halted due to compliance violation: {e}")
        raise
    except Exception as e:
        logger.error(f"Generation halted due to unexpected error: {e}")
        raise

    narratives_dict: dict[str, HubNarrative] = {}
    total_tokens = 0
    summary_words = []
    para_callout_count = 0
    hot_spot_count = sum(1 for hub in hubs if hub.get("is_paralympic_hot_spot"))

    for narrative, tokens in results:
        narratives_dict[narrative.hub_id] = narrative
        total_tokens += tokens
        summary_words.append(len(narrative.summary.split()))
        if narrative.paralympic_callout:
            para_callout_count += 1

    avg_summary_words = sum(summary_words) / len(summary_words) if summary_words else 0

    logger.info(f"{len(results)}/{len(hubs)} narratives passed compliance")
    logger.info(f"Total tokens used: {total_tokens}")
    logger.info(f"Average summary length: {avg_summary_words:.1f} words")
    logger.info(f"Narratives with paralympic_callout: {para_callout_count}")
    logger.info(f"Input hubs flagged as is_paralympic_hot_spot: {hot_spot_count}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    serialized_data = {
        hub_id: narrative.model_dump(mode="json")
        for hub_id, narrative in narratives_dict.items()
    }
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully wrote {len(narratives_dict)} narratives to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

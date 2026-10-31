
"""
LLM Bird Research Service
-------------------------

Uses:
- Google Gemini Interactions API
- Optional Google Search grounding
- GBIF Species API

Responsibilities:
- Research a bird already identified by WingCLIP.
- Generate a structured bird information profile.
- Extract clickable source citations when Search is enabled.
- Cross-check taxonomy using GBIF.
- Cache successful results for 24 hours.

Important:
Gemini does NOT identify the bird in the uploaded photograph.
WingCLIP performs image identification. Gemini only researches
the common name and scientific name returned by WingCLIP.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
).strip()

# Keep false for the free API tier.
# Google Search grounding for Gemini 3.8 Flash is not
# available on the free Gemini API tier.
GEMINI_ENABLE_SEARCH = (
    os.getenv("GEMINI_ENABLE_SEARCH", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

GBIF_API_URL = "https://api.gbif.org/v1"

CACHE_DURATION = 24 * 60 * 60

logger = logging.getLogger("bird-llm-service")


# ============================================================
# STRUCTURED RESPONSE SCHEMA
# ============================================================

BIRD_PROFILE_SCHEMA = {
    "type": "object",

    "properties": {
        "description": {
            "type": "string",
            "description": "A short introduction to this bird."
        },

        "habitat": {
            "type": "string",
            "description": "Habitats where this bird is found."
        },

        "diet": {
            "type": "string",
            "description": "Main foods eaten by this species."
        },

        "distribution": {
            "type": "string",
            "description": "Distribution of this species, emphasizing India."
        },

        "behaviour": {
            "type": "string",
            "description": "Typical behaviour, feeding, nesting or migration."
        },

        "identification_features": {
            "type": "string",
            "description": "Physical features useful for identifying the species."
        },

        "conservation_status": {
            "type": "string",
            "description": "Verified IUCN conservation category, if known."
        },

        "conservation_note": {
            "type": "string",
            "description": "Known threats or population trends, if established."
        },

        "family": {
            "type": "string",
            "description": "Scientific taxonomic family."
        }
    },

    "required": [
        "description",
        "habitat",
        "diet",
        "distribution",
        "behaviour",
        "identification_features",
        "conservation_status",
        "conservation_note",
        "family"
    ]
}


# ============================================================
# CACHE
# ============================================================

_cache: dict[str, dict[str, Any]] = {}

_cache_lock = asyncio.Lock()


async def get_cached_result(
    key: str
) -> dict[str, Any] | None:

    async with _cache_lock:

        entry = _cache.get(key)

        if not entry:
            return None

        if time.monotonic() - entry["time"] > CACHE_DURATION:
            _cache.pop(key, None)
            return None

        result = entry["result"].copy()
        result["cached"] = True

        return result


async def save_cached_result(
    key: str,
    result: dict[str, Any]
) -> None:

    async with _cache_lock:

        _cache[key] = {
            "time": time.monotonic(),
            "result": result.copy()
        }


# ============================================================
# UNAVAILABLE / FALLBACK RESULT
# ============================================================

def research_unavailable(
    status: str,
    message: str
) -> dict[str, Any]:

    return {
        "available": False,
        "research_status": status,
        "provider": "Google Gemini API",
        "model": GEMINI_MODEL,
        "message": message,
        "sources": [],
        "source_name": None,
        "source_url": None,
        "retrieved_at": None,
        "cached": False
    }


# ============================================================
# GBIF TAXONOMY LOOKUP
# ============================================================

async def lookup_gbif_taxonomy(
    scientific_name: str
) -> dict[str, Any] | None:

    """
    Match the scientific name against the GBIF taxonomy.

    GBIF checks taxonomy only. It does not prove that
    WingCLIP's photograph identification is correct.
    """

    url = f"{GBIF_API_URL}/species/match"

    params = {
        "name": scientific_name,
        "strict": "true"
    }

    headers = {
        "User-Agent": "IndianBirdAtlas/1.0"
    }

    try:

        async with httpx.AsyncClient(
            timeout=12.0,
            headers=headers
        ) as client:

            response = await client.get(
                url,
                params=params
            )

            response.raise_for_status()

            data = response.json()

        usage_key = data.get("usageKey")

        if not usage_key or data.get("matchType") == "NONE":
            return None

        return {
            "scientific_name": data.get("scientificName"),
            "canonical_name": data.get("canonicalName"),
            "rank": data.get("rank"),
            "status": data.get("status"),
            "kingdom": data.get("kingdom"),
            "phylum": data.get("phylum"),
            "class": data.get("class"),
            "order": data.get("order"),
            "family": data.get("family"),
            "genus": data.get("genus"),
            "usage_key": usage_key,
            "match_type": data.get("matchType")
        }

    except Exception as exc:

        logger.warning(
            "GBIF taxonomy lookup failed: %s",
            exc
        )

        return None


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def _get_value(
    obj: Any,
    name: str,
    default: Any = None
) -> Any:

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def extract_web_sources(
    response: Any,
    taxonomy: dict[str, Any] | None = None
) -> list[dict[str, str]]:

    """
    Extract inline URL citations from Gemini Search grounding.

    Gemini Interactions API returns citations in the
    annotations attached to model-output text blocks.
    """

    sources: dict[str, dict[str, str]] = {}

    def add_source(
        title: str | None,
        url: str | None
    ) -> None:

        if not url or not url.startswith(("https://", "http://")):
            return

        sources[url] = {
            "title": (
                str(title).strip()
                if title
                else url.split("/")[2]
            ),
            "url": url
        }

    steps = _get_value(response, "steps", []) or []

    for step in steps:

        if _get_value(step, "type") != "model_output":
            continue

        content_blocks = _get_value(step, "content", []) or []

        for content in content_blocks:

            annotations = (
                _get_value(content, "annotations", []) or []
            )

            for annotation in annotations:

                url = (
                    _get_value(annotation, "uri")
                    or _get_value(annotation, "url")
                )

                title = _get_value(annotation, "title")

                add_source(title, url)

    # Include a GBIF reference when taxonomy was matched.
    if taxonomy and taxonomy.get("usage_key"):

        gbif_url = (
            "https://www.gbif.org/species/"
            f"{taxonomy['usage_key']}"
        )

        add_source(
            "GBIF Backbone Taxonomy",
            gbif_url
        )

    return list(sources.values())[:10]


# ============================================================
# RESEARCH PROMPT
# ============================================================

def build_research_prompt(
    common_name: str,
    scientific_name: str,
    ebird_code: str | None,
    taxonomy: dict[str, Any] | None,
    search_enabled: bool
) -> str:

    """
    Scientific name is the primary species identifier.
    The prompt never asks Gemini to identify a photograph.
    """

    taxonomy_text = json.dumps(
        taxonomy or {},
        ensure_ascii=False
    )

    if search_enabled:

        research_instruction = """
Use Google Search grounding to research this species.
Prefer authoritative bird and biodiversity sources.
Base factual claims on the information retrieved.
"""

    else:

        research_instruction = """
Prepare the profile using established ornithological
knowledge and the supplied GBIF taxonomy information.
Live web search is not enabled for this request.
Do not claim that facts have been verified through live
web research. Mark uncertain details as
"Not reliably reported".
"""

    return f"""
You are an ornithology research assistant for an
Indian Bird Atlas.

The image-classification model has ALREADY identified
the following species:

Common name: {common_name}
Scientific name: {scientific_name}
eBird code: {ebird_code or "Not supplied"}

GBIF TAXONOMY CROSS-CHECK:
{taxonomy_text}

IMPORTANT:

1. The scientific name is the primary species identifier.
2. Do not substitute another species with a similar name.
3. Do not identify or reinterpret an uploaded photograph.
4. The bird was identified by WingCLIP, not by you.
5. Focus on reliable information about this species in India.
6. Distinguish resident, breeding, passage migrant and
   winter visitor status only when supported by evidence.
7. Do not invent facts, population figures or dates.
8. If a detail is uncertain, write "Not reliably reported".
9. Give an IUCN category only when reliably established.
10. Do not confuse a global conservation assessment with
    a local Indian population assessment.

{research_instruction}

Create concise information suitable for a student-friendly
bird identification website.

Return these exact fields:

description:
A useful introduction, approximately 50-80 words.

habitat:
Where this bird lives.

diet:
What this bird eats.

distribution:
Its distribution, emphasizing India.

behaviour:
Typical feeding, nesting, social or migration behaviour.

identification_features:
Physical features that distinguish this species.

conservation_status:
Verified IUCN/global status, or
"Not reliably reported".

conservation_note:
Threats or population trends, only when supported.

family:
Scientific family name.

Return the structured JSON requested by the API.
Do not add Markdown or extra fields.
"""


# ============================================================
# GEMINI REQUEST
# ============================================================

def call_gemini(
    prompt: str
) -> tuple[str, Any]:

    """
    Synchronous Gemini call.

    It is executed with asyncio.to_thread() below so that
    FastAPI's event loop is not blocked.
    """

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from backend/.env"
        )

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    try:

        request: dict[str, Any] = {
            "model": GEMINI_MODEL,
            "input": prompt,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": BIRD_PROFILE_SCHEMA
            },
            "store": False
        }

        if GEMINI_ENABLE_SEARCH:
            request["tools"] = [
                {"type": "google_search"}
            ]

        interaction = client.interactions.create(**request)

        output_text = _get_value(
            interaction,
            "output_text",
            ""
        )

        if not output_text or not output_text.strip():
            raise ValueError(
                "Gemini returned an empty research response."
            )

        return output_text, interaction

    finally:
        close_method = getattr(client, "close", None)

        if callable(close_method):
            try:
                close_method()
            except Exception:
                logger.debug(
                    "Gemini client cleanup failed.",
                    exc_info=True
                )


# ============================================================
# MAIN BIRD RESEARCH FUNCTION
# ============================================================

async def research_bird(
    common_name: str,
    scientific_name: str,
    ebird_code: str | None = None
) -> dict[str, Any]:

    """
    Research a bird already identified by WingCLIP.

    The uploaded photograph is NOT sent to Gemini.
    Only the species names and optional eBird code are used.
    """

    common_name = (common_name or "").strip()

    scientific_name = (scientific_name or "").strip()

    ebird_code = (ebird_code or "").strip() or None

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not common_name or not scientific_name:

        return research_unavailable(
            "invalid_species",
            "A common name and scientific name are required."
        )

    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    if not GEMINI_API_KEY:

        return research_unavailable(
            "not_configured",
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to backend/.env."
        )

    # --------------------------------------------------------
    # Check cache
    # --------------------------------------------------------

    cache_key = (
        f"{GEMINI_MODEL}|"
        f"{scientific_name.casefold()}|"
        f"search={GEMINI_ENABLE_SEARCH}"
    )

    cached = await get_cached_result(cache_key)

    if cached:
        return cached

    try:

        logger.info(
            "Researching bird: %s (%s)",
            common_name,
            scientific_name
        )

        # ----------------------------------------------------
        # GBIF taxonomy lookup
        # ----------------------------------------------------

        taxonomy = await lookup_gbif_taxonomy(
            scientific_name
        )

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt = build_research_prompt(
            common_name=common_name,
            scientific_name=scientific_name,
            ebird_code=ebird_code,
            taxonomy=taxonomy,
            search_enabled=GEMINI_ENABLE_SEARCH
        )

        # ----------------------------------------------------
        # Gemini request
        # ----------------------------------------------------

        raw_text, response = await asyncio.to_thread(
            call_gemini,
            prompt
        )

        # ----------------------------------------------------
        # Parse structured JSON
        # ----------------------------------------------------

        profile = json.loads(raw_text)

        if not isinstance(profile, dict):
            raise ValueError(
                "Gemini returned an invalid profile."
            )

        # ----------------------------------------------------
        # Extract citations
        # ----------------------------------------------------

        sources = extract_web_sources(
            response,
            taxonomy
        )

        # ----------------------------------------------------
        # Prefer GBIF family where available
        # ----------------------------------------------------

        family = profile.get("family") or (
            taxonomy.get("family")
            if taxonomy
            else None
        )

        # ----------------------------------------------------
        # Source labels
        # ----------------------------------------------------

        if GEMINI_ENABLE_SEARCH and len(sources) > (
            1 if taxonomy else 0
        ):

            source_name = "Google Search Grounding + GBIF"

        elif taxonomy:

            source_name = "GBIF taxonomy reference"

        else:

            source_name = "Gemini generated profile"

        source_url = (
            sources[0]["url"]
            if sources
            else None
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        result = {

            "available": True,

            "research_status": "success",

            "provider": (
                "Google Gemini API + Google Search"
                if GEMINI_ENABLE_SEARCH
                else "Google Gemini API"
            ),

            "model": GEMINI_MODEL,

            "description": profile.get(
                "description",
                ""
            ),

            "habitat": profile.get(
                "habitat",
                ""
            ),

            "diet": profile.get(
                "diet",
                ""
            ),

            "distribution": profile.get(
                "distribution",
                ""
            ),

            "behaviour": profile.get(
                "behaviour",
                ""
            ),

            "identification_features": profile.get(
                "identification_features",
                ""
            ),

            "conservation_status": profile.get(
                "conservation_status",
                "Not reliably reported"
            ),

            "conservation_note": profile.get(
                "conservation_note",
                ""
            ),

            "family": family or "Not reliably reported",

            "taxonomy": taxonomy,

            "sources": sources,

            "source_name": source_name,

            "source_url": source_url,

            "retrieved_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "cached": False,

            "search_grounding_enabled": GEMINI_ENABLE_SEARCH

        }

        # ----------------------------------------------------
        # Cache successful results
        # ----------------------------------------------------

        await save_cached_result(
            cache_key,
            result
        )

        logger.info(
            "Bird research completed for %s",
            scientific_name
        )

        return result

    except Exception:

        logger.exception(
            "Gemini bird research failed for %s",
            scientific_name
        )

        return research_unavailable(
            "error",
            "Bird research is temporarily unavailable. "
            "The species may still have been identified "
            "successfully by WingCLIP. Check the backend "
            "terminal for the detailed error."
        )

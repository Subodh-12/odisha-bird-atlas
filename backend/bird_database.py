"""Local species-profile database and offline fallback helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATABASE_PATH = Path(__file__).with_name("species_info.json")


def load_species_database() -> dict[str, dict[str, Any]]:
    """Load local profiles, indexed by normalized scientific name."""
    if not DATABASE_PATH.exists():
        return {}

    try:
        with DATABASE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read species database: {exc}"
        ) from exc

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("species"), dict)
    ):
        raise ValueError(
            "species_info.json must contain a 'species' object."
        )

    profiles = {}

    for scientific_name, profile in data["species"].items():
        if not isinstance(profile, dict):
            continue

        key = str(scientific_name).strip().casefold()

        if key:
            profiles[key] = profile

    return profiles


def build_bird_information(
    common_name: str,
    scientific_name: str,
    ebird_code: str | None,
    database: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return a local profile or a clearly labelled fallback."""

    profile = database.get(
        (scientific_name or "").strip().casefold(),
        {}
    )

    has_profile = bool(profile)

    return {
        "available": has_profile,
        "local_profile_available": has_profile,
        "research_available": False,

        "common_name": common_name,
        "scientific_name": scientific_name,
        "ebird_code": ebird_code,

        "description": profile.get(
            "description",
            (
                f"WingCLIP matched this photograph to {common_name} "
                f"({scientific_name}). A detailed offline profile for "
                "this species has not been added yet."
            ),
        ),

        "habitat": profile.get("habitat", ""),
        "diet": profile.get("diet", ""),
        "distribution": profile.get("distribution", ""),
        "behaviour": profile.get("behaviour", ""),
        "identification_features": profile.get(
            "identification_features", ""
        ),
        "conservation_status": profile.get(
            "conservation_status", ""
        ),
        "conservation_note": profile.get(
            "conservation_note", ""
        ),
        "family": profile.get("family", ""),

        "taxonomy": None,
        "sources": profile.get("sources", []),

        "source_name": profile.get(
            "source_name",
            "Local starter species guide" if has_profile else None,
        ),

        "source_url": profile.get("source_url"),

        "research_status": "not_requested",
        "research_message": (
            "Online research has not been completed."
        ),
    }
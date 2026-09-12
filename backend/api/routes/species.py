"""
Species endpoints.

GET /api/species                    — all known species
GET /api/species/{id}               — single species with stats
GET /api/heard-recently             — recently detected species
GET /api/species-artwork/{stem}     — serve artwork image file
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import config
from backend.api.dependencies import get_db_session, get_repository
from backend.database.repository import DetectionRepository
from backend.database.models import Species
from backend.artwork.static_provider import SUPPORTED_EXTENSIONS

router = APIRouter()

GENERATED_DIR = config.ASSETS_DIR / "generated"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SpeciesResponse(BaseModel):
    id: int
    scientific_name: str
    common_name: str
    artwork_path: Optional[str]
    detection_count: int

    model_config = {"from_attributes": True}


class HeardRecentlyResponse(BaseModel):
    hours: int
    count: int
    species: list[SpeciesResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_artwork_stem(scientific_name: str) -> Optional[str]:
    """
    Return artwork stem if any artwork exists for this species.

    Priority:
    1. Generated cut-out PNG (assets/artwork/generated/<stem>.png)
    2. Static illustration (assets/artwork/<stem>.jpg/png/etc)

    Returns None if no artwork found.
    """
    stem = scientific_name.lower().replace(" ", "_")

    # Generated artwork takes priority
    generated_path = GENERATED_DIR / f"{stem}.png"
    if generated_path.exists():
        return f"generated/{stem}"

    # Fall back to static only if no generated artwork
    for ext in SUPPORTED_EXTENSIONS:
        if (config.ASSETS_DIR / f"{stem}{ext}").exists():
            return stem

    return None


def _species_to_response(
    species: Species,
    session,
    repo: DetectionRepository,
) -> SpeciesResponse:
    count = repo.count_detections(session, species_id=species.id)
    artwork_stem = _find_artwork_stem(species.scientific_name)
    return SpeciesResponse(
        id=species.id,
        scientific_name=species.scientific_name,
        common_name=species.common_name,
        artwork_path=artwork_stem,
        detection_count=count,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/species", response_model=list[SpeciesResponse])
def list_species(
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    species_list = repo.list_species(session)
    return [_species_to_response(s, session, repo) for s in species_list]


@router.get("/species/{species_id}", response_model=SpeciesResponse)
def get_species(
    species_id: int,
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    species = repo.get_species_by_id(session, species_id)
    if species is None:
        raise HTTPException(status_code=404, detail="Species not found")
    return _species_to_response(species, session, repo)


@router.get("/heard-recently", response_model=HeardRecentlyResponse)
def heard_recently(
    hours: int = Query(default=config.HEARD_RECENTLY_HOURS, ge=1, le=168),
    limit: int = Query(default=config.COLLAGE_MAX_SPECIES, ge=1, le=20),
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    species_list = repo.get_recently_heard_species(
        session, hours=hours, limit=limit
    )
    return HeardRecentlyResponse(
        hours=hours,
        count=len(species_list),
        species=[
            _species_to_response(s, session, repo) for s in species_list
        ],
    )


@router.get("/species-artwork/{stem:path}")
def get_species_artwork(stem: str):
    """
    Serve artwork image for a species.

    stem formats:
      "generated/erithacus_rubecula"  → generated PNG cut-out
      "erithacus_rubecula"            → static illustration
    """
    if stem.startswith("generated/"):
        safe_name = stem[len("generated/"):]
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid stem.")
        path = GENERATED_DIR / f"{safe_name}.png"
        if path.exists():
            return FileResponse(str(path), media_type="image/png")
        raise HTTPException(status_code=404, detail=f"No generated artwork for '{safe_name}'.")

    safe_stem = "".join(c for c in stem if c.isalnum() or c == "_")
    if not safe_stem:
        raise HTTPException(status_code=400, detail="Invalid stem.")

    for ext in SUPPORTED_EXTENSIONS:
        candidate = config.ASSETS_DIR / f"{safe_stem}{ext}"
        if candidate.exists():
            media_type = (
                "image/jpeg" if ext in (".jpg", ".jpeg")
                else "image/png" if ext == ".png"
                else "image/webp"
            )
            return FileResponse(str(candidate), media_type=media_type)

    raise HTTPException(status_code=404, detail=f"No artwork found for '{safe_stem}'.")
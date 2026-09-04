"""
Species endpoints.

GET /api/species                — all known species
GET /api/species/{id}           — single species with stats
GET /api/heard-recently         — recently detected species
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import config
from backend.api.dependencies import get_db_session, get_repository
from backend.database.repository import DetectionRepository
from backend.database.models import Species

router = APIRouter()


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

def _species_to_response(
    species: Species,
    session,
    repo: DetectionRepository,
) -> SpeciesResponse:
    count = repo.count_detections(session, species_id=species.id)
    return SpeciesResponse(
        id=species.id,
        scientific_name=species.scientific_name,
        common_name=species.common_name,
        artwork_path=species.artwork_path,
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
    """Return all known species, alphabetically by common name."""
    species_list = repo.list_species(session)
    return [_species_to_response(s, session, repo) for s in species_list]


@router.get("/species/{species_id}", response_model=SpeciesResponse)
def get_species(
    species_id: int,
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """Return a single species by ID with detection count."""
    species = repo.get_species_by_id(session, species_id)
    if species is None:
        raise HTTPException(status_code=404, detail="Species not found")
    return _species_to_response(species, session, repo)


@router.get("/heard-recently", response_model=HeardRecentlyResponse)
def heard_recently(
    hours: int = Query(
        default=config.HEARD_RECENTLY_HOURS, ge=1, le=168
    ),
    limit: int = Query(
        default=config.COLLAGE_MAX_SPECIES, ge=1, le=20
    ),
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """
    Return species detected within the last *hours* hours.
    Ordered by most-recently-detected first.
    """
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
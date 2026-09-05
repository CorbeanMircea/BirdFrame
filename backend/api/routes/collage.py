"""
Collage endpoints.

GET  /api/collage/latest     — serve the latest collage image
POST /api/collage/generate   — generate a new collage on demand
GET  /api/collage/status     — collage metadata (exists, age, size)
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import config
from backend.api.dependencies import get_db_session, get_repository
from backend.database.repository import DetectionRepository
from backend.artwork.static_provider import StaticArtworkProvider
from backend.collage.generator import CollageGenerator, CollageGeneratorError

router = APIRouter()

# Module-level singletons — created once, reused across requests
_provider = StaticArtworkProvider()
_generator = CollageGenerator()

# Track when the last generation ran (UTC timestamp)
_last_generated_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CollageStatusResponse(BaseModel):
    exists: bool
    path: Optional[str]
    size_bytes: Optional[int]
    generated_at: Optional[str]
    age_seconds: Optional[float]


class CollageGenerateResponse(BaseModel):
    success: bool
    path: Optional[str]
    species_count: int
    message: str
    generated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_path() -> Path:
    return config.COLLAGE_DIR / "latest.jpg"


def _build_species_paths(
    session: Session,
    repo: DetectionRepository,
    hours: int,
    limit: int,
) -> dict:
    """
    Query recently heard species and build the species_paths dict
    needed by CollageGenerator.generate().
    """
    recent_species = repo.get_recently_heard_species(
        session, hours=hours, limit=limit
    )

    species_paths = {}
    for species in recent_species:
        path = _provider.get_artwork(species.scientific_name)
        if path is None:
            # Try lowercase lookup (provider indexes by lowercase)
            path = _provider.get_artwork(species.scientific_name.lower())
        species_paths[species.scientific_name] = (
            species.common_name,
            path,
        )

    return species_paths


def _generate_collage_now(
    session: Session,
    repo: DetectionRepository,
    hours: int,
    limit: int,
) -> CollageGenerateResponse:
    """Core generation logic, shared between sync and background paths."""
    global _last_generated_at

    species_paths = _build_species_paths(session, repo, hours, limit)

    if not species_paths:
        return CollageGenerateResponse(
            success=False,
            path=None,
            species_count=0,
            message="No species detected recently — nothing to collage.",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # Filter to species that actually have artwork
    with_artwork = {
        k: v for k, v in species_paths.items()
        if v[1] is not None
    }

    if not with_artwork:
        return CollageGenerateResponse(
            success=False,
            path=None,
            species_count=len(species_paths),
            message=(
                f"{len(species_paths)} species detected but none have artwork. "
                "Add illustrations to assets/artwork/ to enable collage generation."
            ),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    output_path = _generator.generate_latest(species_paths)
    _last_generated_at = time.time()

    return CollageGenerateResponse(
        success=True,
        path=str(output_path),
        species_count=len(with_artwork),
        message=f"Collage generated with {len(with_artwork)} species.",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/collage/status", response_model=CollageStatusResponse)
def collage_status():
    """
    Return metadata about the latest collage file.
    Does not generate a new one.
    """
    path = _latest_path()

    if not path.exists():
        return CollageStatusResponse(
            exists=False,
            path=None,
            size_bytes=None,
            generated_at=None,
            age_seconds=None,
        )

    stat = path.stat()
    mtime = stat.st_mtime
    age = time.time() - mtime

    return CollageStatusResponse(
        exists=True,
        path=str(path),
        size_bytes=stat.st_size,
        generated_at=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
        age_seconds=round(age, 1),
    )


@router.get("/collage/latest")
def get_latest_collage():
    """
    Serve the latest collage image as a JPEG file.

    Returns 404 if no collage has been generated yet.
    Generate one first with POST /api/collage/generate.
    """
    path = _latest_path()
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No collage has been generated yet. "
                "POST to /api/collage/generate to create one."
            ),
        )
    return FileResponse(
        str(path),
        media_type="image/jpeg",
        filename="birdframe_collage.jpg",
    )


@router.post("/collage/generate", response_model=CollageGenerateResponse)
def generate_collage(
    hours: int = Query(
        default=config.HEARD_RECENTLY_HOURS,
        ge=1, le=168,
        description="Include species detected within this many hours.",
    ),
    limit: int = Query(
        default=config.COLLAGE_MAX_SPECIES,
        ge=1, le=20,
        description="Maximum number of species to include.",
    ),
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """
    Generate a new collage from recently detected species.

    Queries the database for species detected within *hours* hours,
    retrieves their artwork, and composes a new scattered illustration
    collage. The result overwrites latest.jpg.

    Returns immediately with the result.
    """
    try:
        return _generate_collage_now(session, repo, hours, limit)
    except CollageGeneratorError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Collage generation failed: {exc}",
        )


@router.post("/collage/generate-background", response_model=dict)
def generate_collage_background(
    background_tasks: BackgroundTasks,
    hours: int = Query(default=config.HEARD_RECENTLY_HOURS, ge=1, le=168),
    limit: int = Query(default=config.COLLAGE_MAX_SPECIES, ge=1, le=20),
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """
    Trigger collage generation as a background task.

    Returns immediately with 202 Accepted. The collage is generated
    asynchronously — poll GET /api/collage/status to check when it
    is ready.
    """
    # Build species paths synchronously (DB query is fast)
    species_paths = _build_species_paths(session, repo, hours, limit)

    def _run():
        global _last_generated_at
        try:
            if species_paths:
                _generator.generate_latest(species_paths)
                _last_generated_at = time.time()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "Background collage generation failed: %s", exc
            )

    background_tasks.add_task(_run)

    return {
        "accepted": True,
        "message": "Collage generation started in background.",
        "species_queued": len(species_paths),
    }
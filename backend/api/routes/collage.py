"""
Collage endpoints.

GET  /api/collage/latest            — serve the latest collage image
POST /api/collage/generate          — generate a new collage on demand
GET  /api/collage/status            — collage metadata
POST /api/collage/generate-species  — pre-generate artwork for a species
"""

import sys
import time
import logging
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
from backend.artwork.provider_factory import get_artwork_provider
from backend.collage.generator import CollageGenerator, CollageGeneratorError

router = APIRouter()
logger = logging.getLogger(__name__)

_generator = CollageGenerator()
_last_generated_at: Optional[float] = None

GENERATED_DIR = config.ASSETS_DIR / "generated"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CollageStatusResponse(BaseModel):
    exists: bool
    path: Optional[str]
    size_bytes: Optional[int]
    generated_at: Optional[str]
    age_seconds: Optional[float]
    artwork_backend: str


class CollageGenerateResponse(BaseModel):
    success: bool
    path: Optional[str]
    species_count: int
    species_with_artwork: int
    message: str
    generated_at: str
    artwork_backend: str


class SpeciesArtworkResponse(BaseModel):
    scientific_name: str
    common_name: str
    success: bool
    cached: bool
    path: Optional[str]
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_path() -> Path:
    return config.COLLAGE_DIR / "latest.jpg"


def _get_artwork_for_species(scientific_name: str, common_name: str) -> Optional[Path]:
    """
    Get artwork for a species, checking generated directory first.
    Falls back to provider if not cached.
    """
    stem = scientific_name.lower().replace(" ", "_")

    # Check generated cache first (fastest path — no provider needed)
    generated_path = GENERATED_DIR / f"{stem}.png"
    if generated_path.exists():
        return generated_path

    # Try provider (handles static artwork)
    backend = config.ARTWORK_BACKEND
    provider = get_artwork_provider()
    try:
        if backend == "generated":
            return provider.get_artwork(scientific_name, common_name=common_name)
        else:
            return provider.get_artwork(scientific_name)
    except Exception as exc:
        logger.warning("Could not get artwork for %s: %s", scientific_name, exc)
        return None


def _generate_collage_now(
    session,
    repo: DetectionRepository,
    hours: int,
    limit: int,
) -> CollageGenerateResponse:
    global _last_generated_at
    backend = config.ARTWORK_BACKEND

    recent_species = repo.get_recently_heard_species(
        session, hours=hours, limit=limit
    )

    if not recent_species:
        return CollageGenerateResponse(
            success=False,
            path=None,
            species_count=0,
            species_with_artwork=0,
            message="No species detected recently.",
            generated_at=datetime.now(timezone.utc).isoformat(),
            artwork_backend=backend,
        )

    # Build species_paths using cached artwork only (don't generate new here)
    species_paths = {}
    for species in recent_species:
        path = _get_artwork_for_species(
            species.scientific_name, species.common_name
        )
        species_paths[species.scientific_name] = (species.common_name, path)
        if path:
            logger.debug("Artwork found for %s: %s", species.scientific_name, path)
        else:
            logger.debug("No artwork for %s", species.scientific_name)

    with_artwork = {k: v for k, v in species_paths.items() if v[1] is not None}

    if not with_artwork:
        return CollageGenerateResponse(
            success=False,
            path=None,
            species_count=len(species_paths),
            species_with_artwork=0,
            message=(
                f"{len(species_paths)} species detected but none have artwork yet. "
                "Artwork is being generated in the background — try again in a minute."
            ),
            generated_at=datetime.now(timezone.utc).isoformat(),
            artwork_backend=backend,
        )

    output_path = _generator.generate_latest(species_paths)
    _last_generated_at = time.time()

    return CollageGenerateResponse(
        success=True,
        path=str(output_path),
        species_count=len(species_paths),
        species_with_artwork=len(with_artwork),
        message=f"Collage generated with {len(with_artwork)} of {len(species_paths)} species.",
        generated_at=datetime.now(timezone.utc).isoformat(),
        artwork_backend=backend,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/collage/status", response_model=CollageStatusResponse)
def collage_status():
    path = _latest_path()
    if not path.exists():
        return CollageStatusResponse(
            exists=False, path=None, size_bytes=None,
            generated_at=None, age_seconds=None,
            artwork_backend=config.ARTWORK_BACKEND,
        )
    stat = path.stat()
    return CollageStatusResponse(
        exists=True,
        path=str(path),
        size_bytes=stat.st_size,
        generated_at=datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        age_seconds=round(time.time() - stat.st_mtime, 1),
        artwork_backend=config.ARTWORK_BACKEND,
    )


@router.get("/collage/latest")
def get_latest_collage():
    path = _latest_path()
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No collage yet. POST to /api/collage/generate.",
        )
    return FileResponse(
        str(path), media_type="image/jpeg",
        filename="birdframe_collage.jpg",
    )


@router.post("/collage/generate", response_model=CollageGenerateResponse)
def generate_collage(
    hours: int = Query(default=config.HEARD_RECENTLY_HOURS, ge=1, le=168),
    limit: int = Query(default=config.COLLAGE_MAX_SPECIES, ge=1, le=20),
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """
    Generate a collage from recently detected species.
    Uses cached artwork only — generation happens automatically in background.
    """
    try:
        return _generate_collage_now(session, repo, hours, limit)
    except CollageGeneratorError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.error("Collage generation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")


@router.post("/collage/generate-species", response_model=SpeciesArtworkResponse)
def generate_species_artwork(
    scientific_name: str = Query(...),
    common_name: str = Query(...),
):
    """Pre-generate and cache artwork for a single species."""
    backend = config.ARTWORK_BACKEND
    stem = scientific_name.lower().replace(" ", "_")

    # Check if already cached
    generated_path = GENERATED_DIR / f"{stem}.png"
    was_cached = generated_path.exists()

    try:
        provider = get_artwork_provider()
        if backend == "generated":
            path = provider.get_artwork(scientific_name, common_name=common_name)
        else:
            path = provider.get_artwork(scientific_name)
    except Exception as exc:
        return SpeciesArtworkResponse(
            scientific_name=scientific_name, common_name=common_name,
            success=False, cached=was_cached, path=None, message=str(exc),
        )

    if path is None:
        return SpeciesArtworkResponse(
            scientific_name=scientific_name, common_name=common_name,
            success=False, cached=was_cached, path=None,
            message="No artwork available.",
        )

    return SpeciesArtworkResponse(
        scientific_name=scientific_name, common_name=common_name,
        success=True, cached=was_cached, path=str(path),
        message="Already cached." if was_cached else "Generated and cached.",
    )


@router.post("/collage/generate-background", response_model=dict)
def generate_collage_background(
    background_tasks: BackgroundTasks,
    hours: int = Query(default=config.HEARD_RECENTLY_HOURS, ge=1, le=168),
    limit: int = Query(default=config.COLLAGE_MAX_SPECIES, ge=1, le=20),
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """Trigger collage generation as a background task."""
    recent_species = repo.get_recently_heard_species(
        session, hours=hours, limit=limit
    )
    species_snapshot = [
        (s.scientific_name, s.common_name) for s in recent_species
    ]

    def _run():
        global _last_generated_at
        try:
            species_paths = {}
            for sci, common in species_snapshot:
                path = _get_artwork_for_species(sci, common)
                species_paths[sci] = (common, path)
            if any(v[1] for v in species_paths.values()):
                _generator.generate_latest(species_paths)
                _last_generated_at = time.time()
        except Exception as exc:
            logger.error("Background collage error: %s", exc)

    background_tasks.add_task(_run)
    return {
        "accepted": True,
        "message": "Collage generation started.",
        "species_queued": len(species_snapshot),
        "artwork_backend": config.ARTWORK_BACKEND,
    }
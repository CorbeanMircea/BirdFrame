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

# Module-level singletons
_generator = CollageGenerator()
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


def _build_species_paths(
    session: Session,
    repo: DetectionRepository,
    hours: int,
    limit: int,
) -> dict:
    """Query recently heard species and build species_paths dict."""
    recent_species = repo.get_recently_heard_species(
        session, hours=hours, limit=limit
    )
    provider = get_artwork_provider()

    species_paths = {}
    for species in recent_species:
        # Try scientific name lookup (provider indexes by lowercase)
        path = provider.get_artwork(
            species.scientific_name,
            # Pass common_name for GeneratedArtworkProvider
            **({"common_name": species.common_name}
               if hasattr(provider.get_artwork, "__code__")
               and "common_name" in provider.get_artwork.__code__.co_varnames
               else {})
        )
        species_paths[species.scientific_name] = (
            species.common_name, path
        )

    return species_paths


def _generate_collage_now(
    session: Session,
    repo: DetectionRepository,
    hours: int,
    limit: int,
) -> CollageGenerateResponse:
    """Core generation logic."""
    global _last_generated_at
    backend = config.ARTWORK_BACKEND
    provider = get_artwork_provider()

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

    # Build species_paths — call get_artwork with common_name for generated provider
    species_paths = {}
    for species in recent_species:
        if backend == "generated":
            path = provider.get_artwork(
                species.scientific_name,
                common_name=species.common_name,
            )
        else:
            path = provider.get_artwork(species.scientific_name)
        species_paths[species.scientific_name] = (species.common_name, path)

    with_artwork = {k: v for k, v in species_paths.items() if v[1] is not None}

    if not with_artwork:
        return CollageGenerateResponse(
            success=False,
            path=None,
            species_count=len(species_paths),
            species_with_artwork=0,
            message=(
                f"{len(species_paths)} species detected but none have artwork. "
                "For generated artwork, ensure ComfyUI is running."
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
        message=f"Collage generated with {len(with_artwork)} species.",
        generated_at=datetime.now(timezone.utc).isoformat(),
        artwork_backend=backend,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/collage/status", response_model=CollageStatusResponse)
def collage_status():
    """Return metadata about the latest collage file."""
    path = _latest_path()

    if not path.exists():
        return CollageStatusResponse(
            exists=False,
            path=None,
            size_bytes=None,
            generated_at=None,
            age_seconds=None,
            artwork_backend=config.ARTWORK_BACKEND,
        )

    stat = path.stat()
    mtime = stat.st_mtime
    age = time.time() - mtime

    return CollageStatusResponse(
        exists=True,
        path=str(path),
        size_bytes=stat.st_size,
        generated_at=datetime.fromtimestamp(
            mtime, tz=timezone.utc
        ).isoformat(),
        age_seconds=round(age, 1),
        artwork_backend=config.ARTWORK_BACKEND,
    )


@router.get("/collage/latest")
def get_latest_collage():
    """Serve the latest collage image as a JPEG file."""
    path = _latest_path()
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No collage generated yet. POST to /api/collage/generate.",
        )
    return FileResponse(
        str(path),
        media_type="image/jpeg",
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
    Generate a new collage from recently detected species.

    Uses the configured artwork backend (static or generated).
    With generated backend, new species trigger ComfyUI generation
    (~15-50s per species, cached after first generation).
    """
    try:
        return _generate_collage_now(session, repo, hours, limit)
    except CollageGeneratorError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.error("Collage generation error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Collage generation failed: {exc}",
        )


@router.post(
    "/collage/generate-species",
    response_model=SpeciesArtworkResponse,
)
def generate_species_artwork(
    scientific_name: str = Query(..., description="e.g. Erithacus rubecula"),
    common_name: str = Query(..., description="e.g. European Robin"),
):
    """
    Pre-generate and cache artwork for a single species.

    Useful for warming up the cache before the collage is needed.
    With the static backend this just checks if artwork exists.
    With the generated backend this triggers ComfyUI generation.
    """
    backend = config.ARTWORK_BACKEND
    provider = get_artwork_provider()

    was_cached = provider.has_artwork(scientific_name)

    try:
        if backend == "generated":
            path = provider.get_artwork(
                scientific_name, common_name=common_name
            )
        else:
            path = provider.get_artwork(scientific_name)
    except Exception as exc:
        return SpeciesArtworkResponse(
            scientific_name=scientific_name,
            common_name=common_name,
            success=False,
            cached=was_cached,
            path=None,
            message=str(exc),
        )

    if path is None:
        return SpeciesArtworkResponse(
            scientific_name=scientific_name,
            common_name=common_name,
            success=False,
            cached=was_cached,
            path=None,
            message="No artwork available for this species.",
        )

    return SpeciesArtworkResponse(
        scientific_name=scientific_name,
        common_name=common_name,
        success=True,
        cached=was_cached,
        path=str(path),
        message="Cached." if was_cached else "Generated and cached.",
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
    Returns immediately — poll /api/collage/status to check when ready.
    """
    backend = config.ARTWORK_BACKEND
    recent_species = repo.get_recently_heard_species(
        session, hours=hours, limit=limit
    )

    species_snapshot = [
        (s.scientific_name, s.common_name) for s in recent_species
    ]

    def _run():
        global _last_generated_at
        try:
            provider = get_artwork_provider()
            species_paths = {}
            for sci, common in species_snapshot:
                if backend == "generated":
                    path = provider.get_artwork(sci, common_name=common)
                else:
                    path = provider.get_artwork(sci)
                species_paths[sci] = (common, path)
            if species_paths:
                _generator.generate_latest(species_paths)
                _last_generated_at = time.time()
        except Exception as exc:
            logger.error("Background collage error: %s", exc)

    background_tasks.add_task(_run)

    return {
        "accepted": True,
        "message": "Collage generation started in background.",
        "species_queued": len(species_snapshot),
        "artwork_backend": backend,
    }
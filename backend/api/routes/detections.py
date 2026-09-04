"""
Detection endpoints.

GET /api/detections          — paginated list of detections
GET /api/detections/{id}     — single detection by ID
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.api.dependencies import get_db_session, get_repository
from backend.database.repository import DetectionRepository
from backend.database.models import Detection

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class DetectionResponse(BaseModel):
    id: int
    species_id: int
    scientific_name: str
    common_name: str
    confidence: float
    timestamp: str
    duration_seconds: Optional[float]
    model_name: Optional[str]
    model_version: Optional[str]
    grouped_event_id: Optional[int]

    model_config = {"from_attributes": True}


class DetectionListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[DetectionResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detection_to_response(detection: Detection) -> DetectionResponse:
    return DetectionResponse(
        id=detection.id,
        species_id=detection.species_id,
        scientific_name=detection.species.scientific_name,
        common_name=detection.species.common_name,
        confidence=detection.confidence,
        timestamp=detection.timestamp.isoformat(),
        duration_seconds=detection.duration_seconds,
        model_name=detection.model_name,
        model_version=detection.model_version,
        grouped_event_id=detection.grouped_event_id,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/detections", response_model=DetectionListResponse)
def list_detections(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    species_id: Optional[int] = Query(default=None),
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """
    Return a paginated list of detections, newest first.

    Optional filter: species_id
    """
    detections = repo.list_detections(
        session, limit=limit, offset=offset, species_id=species_id
    )
    total = repo.count_detections(session, species_id=species_id)

    return DetectionListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[_detection_to_response(d) for d in detections],
    )


@router.get("/detections/{detection_id}", response_model=DetectionResponse)
def get_detection(
    detection_id: int,
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """Return a single detection by ID."""
    detection = repo.get_detection_by_id(session, detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    return _detection_to_response(detection)
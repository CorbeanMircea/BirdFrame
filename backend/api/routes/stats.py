"""
Statistics endpoint.

GET /api/stats   — summary statistics
"""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.api.dependencies import get_db_session, get_repository
from backend.database.repository import DetectionRepository

router = APIRouter()


class StatsResponse(BaseModel):
    total_detections: int
    total_species: int
    total_events: int
    latest_detection_at: Optional[str]


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """Return summary statistics for the BirdFrame database."""
    raw = repo.get_stats(session)
    return StatsResponse(
        total_detections=raw["total_detections"],
        total_species=raw["total_species"],
        total_events=raw["total_events"],
        latest_detection_at=raw["latest_detection_at"],
    )
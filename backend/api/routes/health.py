"""
Health and status endpoints.

GET /health        — basic liveness check
GET /api/status    — detailed system status
"""

import sys
import platform
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import config
from backend.api.dependencies import get_db_session, get_repository
from backend.database.repository import DetectionRepository

router = APIRouter()


@router.get("/health")
def health_check():
    """
    Liveness probe. Returns 200 if the API is running.
    Used by monitoring tools and load balancers.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/status")
def system_status(
    session: Session = Depends(get_db_session),
    repo: DetectionRepository = Depends(get_repository),
):
    """
    Detailed system status including database statistics.
    """
    db_stats = repo.get_stats(session)

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
        "python": platform.python_version(),
        "platform": platform.system(),
        "config": {
            "identifier_backend": config.IDENTIFIER_BACKEND,
            "heard_recently_hours": config.HEARD_RECENTLY_HOURS,
            "grouping_gap_seconds": config.GROUPING_GAP_SECONDS,
            "audio_sample_rate": config.AUDIO_SAMPLE_RATE,
            "audio_chunk_duration": config.AUDIO_CHUNK_DURATION,
        },
        "database": db_stats,
    }
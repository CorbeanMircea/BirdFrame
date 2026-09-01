"""
DetectionRepository — all database read/write operations for BirdFrame.

This is the single place the rest of the application talks to the database.
Nothing outside this module should import SQLAlchemy models directly or
construct queries; everything goes through these methods.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import desc, func

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.database.models import Species, Detection, DetectionEvent


class DetectionRepository:
    """
    Data-access layer for BirdFrame.

    Every method takes an explicit `session` argument so the caller
    controls transaction boundaries.  This makes the repository easy
    to test with an in-memory database and keeps it decoupled from
    any particular session-management strategy.

    Typical usage:

        from backend.database.engine import get_session
        from backend.database.repository import DetectionRepository

        repo = DetectionRepository()
        with get_session() as session:
            species = repo.get_or_create_species(
                session, "Erithacus rubecula", "European Robin"
            )
            session.commit()
    """

    # ------------------------------------------------------------------
    # Species
    # ------------------------------------------------------------------

    def get_or_create_species(
        self,
        session: Session,
        scientific_name: str,
        common_name: str,
    ) -> Species:
        """
        Return the Species row for *scientific_name*, creating it if needed.

        This is the standard entry-point when a new identification result
        arrives: we may or may not have seen this species before.
        """
        species = (
            session.query(Species)
            .filter(Species.scientific_name == scientific_name)
            .first()
        )
        if species is None:
            species = Species(
                scientific_name=scientific_name,
                common_name=common_name,
            )
            session.add(species)
            session.flush()  # populate species.id without committing
        return species

    def get_species_by_id(
        self, session: Session, species_id: int
    ) -> Optional[Species]:
        """Return a Species by primary key, or None."""
        return session.query(Species).filter(Species.id == species_id).first()

    def get_species_by_scientific_name(
        self, session: Session, scientific_name: str
    ) -> Optional[Species]:
        """Return a Species by scientific name, or None."""
        return (
            session.query(Species)
            .filter(Species.scientific_name == scientific_name)
            .first()
        )

    def list_species(self, session: Session) -> list[Species]:
        """Return all known species, ordered alphabetically by common name."""
        return (
            session.query(Species)
            .order_by(Species.common_name)
            .all()
        )

    def update_species_artwork(
        self, session: Session, species_id: int, artwork_path: str
    ) -> Optional[Species]:
        """Set the artwork_path for a species. Returns the updated row or None."""
        species = self.get_species_by_id(session, species_id)
        if species is None:
            return None
        species.artwork_path = artwork_path
        session.flush()
        return species

    # ------------------------------------------------------------------
    # Detections
    # ------------------------------------------------------------------

    def add_detection(
        self,
        session: Session,
        species_id: int,
        confidence: float,
        timestamp: datetime,
        duration_seconds: Optional[float] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        audio_path: Optional[str] = None,
        grouped_event_id: Optional[int] = None,
    ) -> Detection:
        """
        Persist a single identification result.

        The caller must call session.commit() after this to make the
        write durable.
        """
        detection = Detection(
            species_id=species_id,
            confidence=confidence,
            timestamp=timestamp,
            duration_seconds=duration_seconds,
            model_name=model_name,
            model_version=model_version,
            audio_path=audio_path,
            grouped_event_id=grouped_event_id,
        )
        session.add(detection)
        session.flush()
        return detection

    def get_detection_by_id(
        self, session: Session, detection_id: int
    ) -> Optional[Detection]:
        """Return a Detection by primary key, or None."""
        return (
            session.query(Detection)
            .filter(Detection.id == detection_id)
            .first()
        )

    def list_detections(
        self,
        session: Session,
        limit: int = 50,
        offset: int = 0,
        species_id: Optional[int] = None,
    ) -> list[Detection]:
        """
        Return recent detections, newest first.

        Optionally filter by species_id.
        """
        query = session.query(Detection).order_by(desc(Detection.timestamp))
        if species_id is not None:
            query = query.filter(Detection.species_id == species_id)
        return query.offset(offset).limit(limit).all()

    def count_detections(
        self,
        session: Session,
        species_id: Optional[int] = None,
    ) -> int:
        """Total number of detections, optionally filtered by species."""
        query = session.query(func.count(Detection.id))
        if species_id is not None:
            query = query.filter(Detection.species_id == species_id)
        return query.scalar() or 0

    # ------------------------------------------------------------------
    # "Heard recently" query
    # ------------------------------------------------------------------

    def get_recently_heard_species(
        self,
        session: Session,
        hours: int = config.HEARD_RECENTLY_HOURS,
        limit: int = config.COLLAGE_MAX_SPECIES,
    ) -> list[Species]:
        """
        Return up to *limit* species detected within the last *hours* hours.

        Results are ordered by most-recently-detected first.
        Species are deduplicated — each appears at most once regardless
        of how many detections it has.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Subquery: latest detection timestamp per species within window
        latest_per_species = (
            session.query(
                Detection.species_id,
                func.max(Detection.timestamp).label("latest"),
            )
            .filter(Detection.timestamp >= cutoff)
            .group_by(Detection.species_id)
            .subquery()
        )

        results = (
            session.query(Species)
            .join(latest_per_species, Species.id == latest_per_species.c.species_id)
            .order_by(desc(latest_per_species.c.latest))
            .limit(limit)
            .all()
        )
        return results

    # ------------------------------------------------------------------
    # Detection events
    # ------------------------------------------------------------------

    def add_detection_event(
        self,
        session: Session,
        species_id: int,
        started_at: datetime,
        peak_confidence: float,
        detection_count: int = 1,
        ended_at: Optional[datetime] = None,
    ) -> DetectionEvent:
        """Create and persist a new DetectionEvent."""
        event = DetectionEvent(
            species_id=species_id,
            started_at=started_at,
            ended_at=ended_at,
            peak_confidence=peak_confidence,
            detection_count=detection_count,
        )
        session.add(event)
        session.flush()
        return event

    def get_detection_event_by_id(
        self, session: Session, event_id: int
    ) -> Optional[DetectionEvent]:
        """Return a DetectionEvent by primary key, or None."""
        return (
            session.query(DetectionEvent)
            .filter(DetectionEvent.id == event_id)
            .first()
        )

    def list_detection_events(
        self,
        session: Session,
        limit: int = 50,
        offset: int = 0,
        species_id: Optional[int] = None,
    ) -> list[DetectionEvent]:
        """Return recent detection events, newest first."""
        query = session.query(DetectionEvent).order_by(
            desc(DetectionEvent.started_at)
        )
        if species_id is not None:
            query = query.filter(DetectionEvent.species_id == species_id)
        return query.offset(offset).limit(limit).all()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, session: Session) -> dict:
        """
        Return a summary statistics dictionary for the API /stats endpoint.
        """
        total_detections = self.count_detections(session)
        total_species = session.query(func.count(Species.id)).scalar() or 0
        total_events = (
            session.query(func.count(DetectionEvent.id)).scalar() or 0
        )

        latest_detection = (
            session.query(Detection)
            .order_by(desc(Detection.timestamp))
            .first()
        )

        return {
            "total_detections": total_detections,
            "total_species": total_species,
            "total_events": total_events,
            "latest_detection_at": (
                latest_detection.timestamp.isoformat()
                if latest_detection
                else None
            ),
        }
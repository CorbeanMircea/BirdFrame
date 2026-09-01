"""
SQLAlchemy ORM models for BirdFrame.

These classes define every table in the SQLite database.
Import `Base` and call `Base.metadata.create_all(engine)` to
create the schema.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all BirdFrame models."""
    pass


class Species(Base):
    """
    A bird species that has been detected at least once.

    Scientific name is the stable unique key; common name is
    display-only and may change.
    """

    __tablename__ = "species"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # e.g. "Erithacus rubecula"
    scientific_name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)

    # e.g. "European Robin"
    common_name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Local path to the illustration file, relative to ASSETS_DIR
    # Null until artwork has been sourced for this species.
    artwork_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="species", cascade="all, delete-orphan"
    )
    detection_events: Mapped[list["DetectionEvent"]] = relationship(
        "DetectionEvent", back_populates="species", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Species id={self.id} scientific_name={self.scientific_name!r}>"


class DetectionEvent(Base):
    """
    A grouped detection event: one bird present for a continuous stretch.

    Multiple raw Detections (e.g. the same robin detected in 5 consecutive
    3-second windows) are collapsed into a single DetectionEvent.
    Grouping logic lives in DetectionService / DetectionGrouper (Phase 6).
    """

    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    species_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("species.id"), nullable=False
    )

    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Highest confidence seen across all detections in this event
    peak_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Running count of raw detections merged into this event
    detection_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Relationships
    species: Mapped["Species"] = relationship("Species", back_populates="detection_events")
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="event"
    )

    def __repr__(self) -> str:
        return (
            f"<DetectionEvent id={self.id} species_id={self.species_id} "
            f"started_at={self.started_at}>"
        )


class Detection(Base):
    """
    A single raw detection: one identification result from one audio chunk.

    Many Detections may be grouped into one DetectionEvent, but a Detection
    is always recorded individually so the raw data is never lost.
    """

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    species_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("species.id"), nullable=False
    )

    # FK to DetectionEvent — null until grouping has run (Phase 6)
    grouped_event_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("detection_events.id"), nullable=True
    )

    # 0.0–1.0 as reported by the identifier model
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # When the audio chunk started
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Length of the audio chunk that was classified
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Which model produced this result, e.g. "BirdNET"
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Model version string, e.g. "2.4"
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Path to retained audio clip (null when AUDIO_RETAIN_CLIPS is False)
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    species: Mapped["Species"] = relationship("Species", back_populates="detections")
    event: Mapped["DetectionEvent | None"] = relationship(
        "DetectionEvent", back_populates="detections"
    )

    def __repr__(self) -> str:
        return (
            f"<Detection id={self.id} species_id={self.species_id} "
            f"confidence={self.confidence:.2f} timestamp={self.timestamp}>"
        )
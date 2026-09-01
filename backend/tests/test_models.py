"""
Tests for database models and engine initialisation.

Uses an in-memory SQLite database so the real birdframe.db is never touched.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database.models import Base, Species, Detection, DetectionEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """In-memory SQLite engine — created once for the whole module."""
    _engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(_engine)
    yield _engine
    _engine.dispose()


@pytest.fixture
def session(engine):
    """Fresh session for each test; rolls back after the test."""
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    _session = SessionLocal()
    yield _session
    _session.rollback()
    _session.close()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_all_tables_created(self, engine):
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "species" in tables
        assert "detections" in tables
        assert "detection_events" in tables

    def test_species_columns(self, engine):
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("species")}
        assert {"id", "scientific_name", "common_name", "artwork_path", "created_at"} <= cols

    def test_detections_columns(self, engine):
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("detections")}
        expected = {
            "id", "species_id", "grouped_event_id", "confidence",
            "timestamp", "duration_seconds", "model_name", "model_version", "audio_path",
        }
        assert expected <= cols

    def test_detection_events_columns(self, engine):
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("detection_events")}
        expected = {
            "id", "species_id", "started_at", "ended_at",
            "peak_confidence", "detection_count",
        }
        assert expected <= cols


# ---------------------------------------------------------------------------
# Species model tests
# ---------------------------------------------------------------------------

class TestSpeciesModel:
    def test_create_species(self, session):
        species = Species(
            scientific_name="Erithacus rubecula",
            common_name="European Robin",
        )
        session.add(species)
        session.commit()

        fetched = session.query(Species).filter_by(scientific_name="Erithacus rubecula").first()
        assert fetched is not None
        assert fetched.common_name == "European Robin"
        assert fetched.artwork_path is None

    def test_species_scientific_name_unique(self, session):
        session.add(Species(scientific_name="Parus major", common_name="Great Tit"))
        session.commit()

        session.add(Species(scientific_name="Parus major", common_name="Duplicate"))
        with pytest.raises(Exception):
            session.commit()

    def test_species_repr(self, session):
        s = Species(scientific_name="Turdus merula", common_name="Eurasian Blackbird")
        session.add(s)
        session.commit()
        assert "Turdus merula" in repr(s)


# ---------------------------------------------------------------------------
# Detection model tests
# ---------------------------------------------------------------------------

class TestDetectionModel:
    def _make_species(self, session, scientific="Passer domesticus", common="House Sparrow"):
        s = Species(scientific_name=scientific, common_name=common)
        session.add(s)
        session.commit()
        return s

    def test_create_detection(self, session):
        species = self._make_species(session)
        now = datetime.now(timezone.utc)
        detection = Detection(
            species_id=species.id,
            confidence=0.87,
            timestamp=now,
            duration_seconds=3.0,
            model_name="MockIdentifier",
            model_version="0.1",
        )
        session.add(detection)
        session.commit()

        fetched = session.query(Detection).filter_by(species_id=species.id).first()
        assert fetched is not None
        assert fetched.confidence == pytest.approx(0.87)
        assert fetched.model_name == "MockIdentifier"
        assert fetched.audio_path is None
        assert fetched.grouped_event_id is None

    def test_detection_repr(self, session):
        species = self._make_species(session, "Cyanistes caeruleus", "Blue Tit")
        d = Detection(
            species_id=species.id,
            confidence=0.75,
            timestamp=datetime.now(timezone.utc),
        )
        session.add(d)
        session.commit()
        assert "0.75" in repr(d)

    def test_detection_relationship_to_species(self, session):
        species = self._make_species(session, "Fringilla coelebs", "Common Chaffinch")
        d = Detection(
            species_id=species.id,
            confidence=0.60,
            timestamp=datetime.now(timezone.utc),
        )
        session.add(d)
        session.commit()

        assert d.species.scientific_name == "Fringilla coelebs"


# ---------------------------------------------------------------------------
# DetectionEvent model tests
# ---------------------------------------------------------------------------

class TestDetectionEventModel:
    def test_create_event(self, session):
        species = Species(scientific_name="Sylvia atricapilla", common_name="Eurasian Blackcap")
        session.add(species)
        session.commit()

        now = datetime.now(timezone.utc)
        event = DetectionEvent(
            species_id=species.id,
            started_at=now,
            peak_confidence=0.91,
            detection_count=3,
        )
        session.add(event)
        session.commit()

        fetched = session.query(DetectionEvent).filter_by(species_id=species.id).first()
        assert fetched is not None
        assert fetched.peak_confidence == pytest.approx(0.91)
        assert fetched.detection_count == 3
        assert fetched.ended_at is None

    def test_event_repr(self, session):
        species = Species(scientific_name="Columba palumbus", common_name="Common Wood Pigeon")
        session.add(species)
        session.commit()

        ev = DetectionEvent(
            species_id=species.id,
            started_at=datetime.now(timezone.utc),
            peak_confidence=0.5,
        )
        session.add(ev)
        session.commit()
        assert "DetectionEvent" in repr(ev)

    def test_event_detection_relationship(self, session):
        species = Species(scientific_name="Carduelis carduelis", common_name="European Goldfinch")
        session.add(species)
        session.commit()

        now = datetime.now(timezone.utc)
        ev = DetectionEvent(
            species_id=species.id,
            started_at=now,
            peak_confidence=0.80,
        )
        session.add(ev)
        session.commit()

        d = Detection(
            species_id=species.id,
            grouped_event_id=ev.id,
            confidence=0.80,
            timestamp=now,
        )
        session.add(d)
        session.commit()

        assert len(ev.detections) == 1
        assert ev.detections[0].confidence == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Engine / init_db tests
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_init_db_creates_tables(self):
        """init_db() must be safe to call on an already-initialised database."""
        from backend.database.engine import init_db, engine as real_engine
        # Should not raise even though tables already exist
        init_db()
        inspector = inspect(real_engine)
        assert "species" in inspector.get_table_names()

    def test_wal_mode_enabled(self):
        """SQLite WAL journal mode should be active."""
        from backend.database.engine import engine as real_engine
        with real_engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).fetchone()
            assert result[0].upper() == "WAL"

    def test_foreign_keys_enabled(self):
        """SQLite foreign key enforcement should be on."""
        from backend.database.engine import engine as real_engine
        with real_engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys")).fetchone()
            assert result[0] == 1
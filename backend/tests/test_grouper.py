"""
Tests for DetectionGrouper.

Uses an in-memory SQLite database (shared connection pattern).
"""

import sys
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database.models import Base, Detection, DetectionEvent, Species
from backend.database.repository import DetectionRepository
from backend.detection.grouper import DetectionGrouper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_engine():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    engine = create_engine("sqlite://", creator=lambda: conn)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    conn.close()


@pytest.fixture
def session(mem_engine):
    SessionLocal = sessionmaker(
        bind=mem_engine, autocommit=False, autoflush=False,
        expire_on_commit=False,
    )
    s = SessionLocal()
    yield s
    s.rollback()
    s.close()


@pytest.fixture
def repo():
    return DetectionRepository()


@pytest.fixture
def grouper(repo):
    return DetectionGrouper(repo, gap_seconds=60.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def _add_species(session, repo, scientific="Erithacus rubecula",
                 common="European Robin"):
    s = repo.get_or_create_species(session, scientific, common)
    session.flush()
    return s


def _add_detection(session, repo, species_id, timestamp, confidence=0.8):
    d = repo.add_detection(
        session,
        species_id=species_id,
        confidence=confidence,
        timestamp=timestamp,
        model_name="Mock",
        model_version="0.1",
    )
    session.flush()
    return d


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_gap(self, repo):
        g = DetectionGrouper(repo)
        assert g.gap_seconds > 0

    def test_custom_gap(self, repo):
        g = DetectionGrouper(repo, gap_seconds=30.0)
        assert g.gap_seconds == 30.0

    def test_zero_gap_raises(self, repo):
        with pytest.raises(ValueError, match="positive"):
            DetectionGrouper(repo, gap_seconds=0.0)

    def test_negative_gap_raises(self, repo):
        with pytest.raises(ValueError, match="positive"):
            DetectionGrouper(repo, gap_seconds=-10.0)


# ---------------------------------------------------------------------------
# Core grouping logic tests
# ---------------------------------------------------------------------------

class TestGrouping:
    def test_first_detection_creates_new_event(self, session, repo, grouper):
        species = _add_species(session, repo)
        d = _add_detection(session, repo, species.id, _now())

        event = grouper.process(session, d)
        session.commit()

        assert event.id is not None
        assert event.species_id == species.id
        assert event.detection_count == 1
        assert event.peak_confidence == pytest.approx(0.8)
        assert event.ended_at is None  # single detection

    def test_second_detection_within_gap_extends_event(
        self, session, repo, grouper
    ):
        species = _add_species(session, repo)
        now = _now()

        d1 = _add_detection(session, repo, species.id, now, confidence=0.7)
        event1 = grouper.process(session, d1)
        session.commit()

        d2 = _add_detection(
            session, repo, species.id,
            now + timedelta(seconds=10), confidence=0.9
        )
        event2 = grouper.process(session, d2)
        session.commit()

        assert event1.id == event2.id, "Same event must be reused"
        assert event2.detection_count == 2
        assert event2.peak_confidence == pytest.approx(0.9)

    def test_detection_outside_gap_creates_new_event(
        self, session, repo, grouper
    ):
        species = _add_species(session, repo)
        now = _now()

        d1 = _add_detection(session, repo, species.id, now)
        event1 = grouper.process(session, d1)
        session.commit()

        # 90 seconds later — beyond 60s gap
        d2 = _add_detection(
            session, repo, species.id,
            now + timedelta(seconds=90)
        )
        event2 = grouper.process(session, d2)
        session.commit()

        assert event1.id != event2.id, "New event must be created"
        assert event2.detection_count == 1

    def test_detection_linked_to_event(self, session, repo, grouper):
        species = _add_species(session, repo)
        d = _add_detection(session, repo, species.id, _now())
        event = grouper.process(session, d)
        session.commit()
        assert d.grouped_event_id == event.id

    def test_different_species_get_different_events(
        self, session, repo, grouper
    ):
        now = _now()
        s1 = _add_species(session, repo, "Erithacus rubecula", "European Robin")
        s2 = _add_species(session, repo, "Parus major", "Great Tit")

        d1 = _add_detection(session, repo, s1.id, now)
        d2 = _add_detection(session, repo, s2.id, now)

        e1 = grouper.process(session, d1)
        e2 = grouper.process(session, d2)
        session.commit()

        assert e1.id != e2.id
        assert e1.species_id == s1.id
        assert e2.species_id == s2.id

    def test_peak_confidence_tracks_maximum(self, session, repo, grouper):
        species = _add_species(session, repo)
        now = _now()

        confidences = [0.6, 0.9, 0.75, 0.85]
        event = None
        for i, conf in enumerate(confidences):
            d = _add_detection(
                session, repo, species.id,
                now + timedelta(seconds=i * 3),
                confidence=conf,
            )
            event = grouper.process(session, d)
        session.commit()

        # Re-read from DB to get committed values
        session.refresh(event)
        assert event.peak_confidence == pytest.approx(0.9)

    def test_detection_count_increments(self, session, repo, grouper):
        species = _add_species(session, repo)
        now = _now()

        event = None
        for i in range(5):
            d = _add_detection(
                session, repo, species.id,
                now + timedelta(seconds=i * 3),
            )
            event = grouper.process(session, d)
        session.commit()

        session.refresh(event)
        assert event.detection_count == 5

    def test_unflushed_detection_raises(self, session, repo, grouper):
        d = Detection(
            species_id=1,
            confidence=0.8,
            timestamp=_now(),
        )
        with pytest.raises(ValueError, match="flushed"):
            grouper.process(session, d)

    def test_many_detections_single_event(self, session, repo, grouper):
        """10 detections within gap → single event."""
        species = _add_species(session, repo)
        now = _now()

        first_event_id = None
        event = None
        for i in range(10):
            d = _add_detection(
                session, repo, species.id,
                now + timedelta(seconds=i * 3),
            )
            event = grouper.process(session, d)
            if first_event_id is None:
                first_event_id = event.id
        session.commit()

        session.refresh(event)
        assert event.id == first_event_id
        assert event.detection_count == 10

    def test_gap_exactly_at_boundary_extends(self, session, repo, grouper):
        """Detection exactly at gap boundary should extend existing event."""
        species = _add_species(session, repo)
        now = _now()

        d1 = _add_detection(session, repo, species.id, now)
        e1 = grouper.process(session, d1)

        d2 = _add_detection(
            session, repo, species.id,
            now + timedelta(seconds=grouper.gap_seconds)
        )
        e2 = grouper.process(session, d2)
        session.commit()

        assert e1.id == e2.id


# ---------------------------------------------------------------------------
# close_stale_events tests
# ---------------------------------------------------------------------------

class TestCloseStaleEvents:
    def test_closes_old_single_detection_events(self, session, repo, grouper):
        """Single-detection events older than gap_seconds must be closed."""
        species = _add_species(session, repo)
        old_time = _now() - timedelta(seconds=120)

        d = _add_detection(session, repo, species.id, old_time)
        event = grouper.process(session, d)
        session.commit()

        # Event is single-detection (ended_at=None) and old
        assert event.ended_at is None

        closed = grouper.close_stale_events(session, reference_time=_now())
        session.commit()

        session.refresh(event)
        assert event.ended_at is not None
        assert closed == 1

    def test_does_not_close_recent_events(self, session, repo, grouper):
        species = _add_species(session, repo)
        now = _now()

        d = _add_detection(session, repo, species.id, now)
        grouper.process(session, d)
        session.commit()

        # Reference time is only 1 second after detection — within gap
        closed = grouper.close_stale_events(
            session,
            reference_time=now + timedelta(seconds=1),
        )
        session.commit()
        assert closed == 0

    def test_returns_count_of_closed_events(self, session, repo, grouper):
        now = _now()
        old_time = now - timedelta(seconds=200)

        s1 = _add_species(session, repo, "Erithacus rubecula", "Robin")
        s2 = _add_species(session, repo, "Parus major", "Great Tit")

        d1 = _add_detection(session, repo, s1.id, old_time)
        d2 = _add_detection(session, repo, s2.id, old_time)
        grouper.process(session, d1)
        grouper.process(session, d2)
        session.commit()

        closed = grouper.close_stale_events(session, reference_time=now)
        session.commit()
        assert closed == 2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestGrouperIntegration:
    def test_realistic_bird_visit_scenario(self, session, repo):
        """
        Simulate a realistic bird visit:
        - Robin sings for 30 seconds (10 detections × 3s)
        - 5-minute silence
        - Robin returns for 9 seconds (3 detections)
        → Should produce 2 DetectionEvents
        """
        grouper = DetectionGrouper(repo, gap_seconds=60.0)
        species = _add_species(session, repo)
        now = _now()

        # First visit: 10 detections, 3 seconds apart
        for i in range(10):
            d = _add_detection(
                session, repo, species.id,
                now + timedelta(seconds=i * 3),
                confidence=0.75 + i * 0.01,
            )
            grouper.process(session, d)
        session.commit()

        # 5-minute gap (300s > 60s gap)
        gap_start = now + timedelta(seconds=10 * 3 + 300)

        # Second visit: 3 detections
        for i in range(3):
            d = _add_detection(
                session, repo, species.id,
                gap_start + timedelta(seconds=i * 3),
                confidence=0.8,
            )
            grouper.process(session, d)
        session.commit()

        events = session.query(DetectionEvent).filter(
            DetectionEvent.species_id == species.id
        ).all()

        assert len(events) == 2, f"Expected 2 events, got {len(events)}"

        first_event = min(events, key=lambda e: e.started_at)
        second_event = max(events, key=lambda e: e.started_at)

        assert first_event.detection_count == 10
        assert second_event.detection_count == 3

    def test_all_detections_linked_to_events(self, session, repo):
        """Every detection must have a grouped_event_id after processing."""
        grouper = DetectionGrouper(repo, gap_seconds=60.0)
        species = _add_species(session, repo)
        now = _now()

        detections = []
        for i in range(5):
            d = _add_detection(
                session, repo, species.id,
                now + timedelta(seconds=i * 3),
            )
            grouper.process(session, d)
            detections.append(d)
        session.commit()

        for d in detections:
            session.refresh(d)
            assert d.grouped_event_id is not None
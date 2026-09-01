"""
Tests for DetectionRepository.

All tests use an in-memory SQLite database; the real birdframe.db is
never touched.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database.models import Base
from backend.database.repository import DetectionRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    _engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(_engine)
    yield _engine
    _engine.dispose()


@pytest.fixture
def session(engine):
    """Fresh, isolated session per test — always rolled back."""
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    _session = SessionLocal()
    yield _session
    _session.rollback()
    _session.close()


@pytest.fixture
def repo() -> DetectionRepository:
    return DetectionRepository()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _add_species(repo, session, scientific="Erithacus rubecula", common="European Robin"):
    s = repo.get_or_create_species(session, scientific, common)
    session.commit()
    return s


# ---------------------------------------------------------------------------
# Species tests
# ---------------------------------------------------------------------------

class TestSpeciesCRUD:
    def test_create_new_species(self, repo, session):
        s = repo.get_or_create_species(session, "Parus major", "Great Tit")
        session.commit()
        assert s.id is not None
        assert s.common_name == "Great Tit"

    def test_get_existing_species(self, repo, session):
        repo.get_or_create_species(session, "Turdus merula", "Eurasian Blackbird")
        session.commit()
        # Call again — must return the same row, not a duplicate
        s2 = repo.get_or_create_species(session, "Turdus merula", "Eurasian Blackbird")
        session.commit()
        count = session.query(__import__("backend.database.models", fromlist=["Species"]).Species).filter_by(
            scientific_name="Turdus merula"
        ).count()
        assert count == 1
        assert s2.scientific_name == "Turdus merula"

    def test_get_species_by_id(self, repo, session):
        s = _add_species(repo, session, "Passer domesticus", "House Sparrow")
        fetched = repo.get_species_by_id(session, s.id)
        assert fetched is not None
        assert fetched.scientific_name == "Passer domesticus"

    def test_get_species_by_id_missing(self, repo, session):
        result = repo.get_species_by_id(session, 999999)
        assert result is None

    def test_get_species_by_scientific_name(self, repo, session):
        _add_species(repo, session, "Fringilla coelebs", "Common Chaffinch")
        found = repo.get_species_by_scientific_name(session, "Fringilla coelebs")
        assert found is not None
        assert found.common_name == "Common Chaffinch"

    def test_get_species_by_scientific_name_missing(self, repo, session):
        assert repo.get_species_by_scientific_name(session, "Nope nopensis") is None

    def test_list_species_alphabetical(self, repo, session):
        repo.get_or_create_species(session, "Sylvia atricapilla", "Eurasian Blackcap")
        repo.get_or_create_species(session, "Cyanistes caeruleus", "Blue Tit")
        repo.get_or_create_species(session, "Carduelis carduelis", "European Goldfinch")
        session.commit()
        species_list = repo.list_species(session)
        names = [s.common_name for s in species_list]
        assert names == sorted(names)

    def test_update_species_artwork(self, repo, session):
        s = _add_species(repo, session, "Columba palumbus", "Common Wood Pigeon")
        updated = repo.update_species_artwork(session, s.id, "pigeon.jpg")
        session.commit()
        assert updated.artwork_path == "pigeon.jpg"

    def test_update_species_artwork_missing_species(self, repo, session):
        result = repo.update_species_artwork(session, 999999, "ghost.jpg")
        assert result is None


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetectionCRUD:
    def test_add_detection(self, repo, session):
        s = _add_species(repo, session, "Motacilla alba", "White Wagtail")
        d = repo.add_detection(
            session,
            species_id=s.id,
            confidence=0.82,
            timestamp=_now(),
            duration_seconds=3.0,
            model_name="MockIdentifier",
            model_version="0.1",
        )
        session.commit()
        assert d.id is not None
        assert d.confidence == pytest.approx(0.82)
        assert d.audio_path is None
        assert d.grouped_event_id is None

    def test_get_detection_by_id(self, repo, session):
        s = _add_species(repo, session, "Hirundo rustica", "Barn Swallow")
        d = repo.add_detection(session, s.id, 0.75, _now())
        session.commit()
        fetched = repo.get_detection_by_id(session, d.id)
        assert fetched is not None
        assert fetched.confidence == pytest.approx(0.75)

    def test_get_detection_by_id_missing(self, repo, session):
        assert repo.get_detection_by_id(session, 999999) is None

    def test_list_detections_newest_first(self, repo, session):
        s = _add_species(repo, session, "Apus apus", "Common Swift")
        now = _now()
        repo.add_detection(session, s.id, 0.6, now - timedelta(minutes=10))
        repo.add_detection(session, s.id, 0.7, now - timedelta(minutes=5))
        repo.add_detection(session, s.id, 0.8, now)
        session.commit()

        detections = repo.list_detections(session, species_id=s.id)
        timestamps = [d.timestamp for d in detections]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_detections_limit(self, repo, session):
        s = _add_species(repo, session, "Cuculus canorus", "Common Cuckoo")
        now = _now()
        for i in range(5):
            repo.add_detection(session, s.id, 0.5, now - timedelta(minutes=i))
        session.commit()

        result = repo.list_detections(session, limit=3, species_id=s.id)
        assert len(result) <= 3

    def test_count_detections(self, repo, session):
        s = _add_species(repo, session, "Upupa epops", "Eurasian Hoopoe")
        now = _now()
        repo.add_detection(session, s.id, 0.9, now)
        repo.add_detection(session, s.id, 0.85, now - timedelta(seconds=3))
        session.commit()

        count = repo.count_detections(session, species_id=s.id)
        assert count == 2


# ---------------------------------------------------------------------------
# "Heard recently" tests
# ---------------------------------------------------------------------------

class TestHeardRecently:
    def test_returns_species_within_window(self, repo, session):
        s = _add_species(repo, session, "Oriolus oriolus", "Eurasian Golden Oriole")
        repo.add_detection(session, s.id, 0.9, _now())
        session.commit()

        # Use a high limit so other test data doesn't push this species out
        recent = repo.get_recently_heard_species(session, hours=24, limit=100)
        scientific_names = [sp.scientific_name for sp in recent]
        assert "Oriolus oriolus" in scientific_names

    def test_excludes_species_outside_window(self, repo, session):
        s = _add_species(repo, session, "Ciconia ciconia", "White Stork")
        old_time = _now() - timedelta(hours=48)
        repo.add_detection(session, s.id, 0.9, old_time)
        session.commit()

        recent = repo.get_recently_heard_species(session, hours=24, limit=100)
        scientific_names = [sp.scientific_name for sp in recent]
        assert "Ciconia ciconia" not in scientific_names

    def test_species_not_duplicated(self, repo, session):
        s = _add_species(repo, session, "Alcedo atthis", "Common Kingfisher")
        now = _now()
        # Three detections of the same species
        repo.add_detection(session, s.id, 0.8, now)
        repo.add_detection(session, s.id, 0.85, now - timedelta(seconds=3))
        repo.add_detection(session, s.id, 0.9, now - timedelta(seconds=6))
        session.commit()

        recent = repo.get_recently_heard_species(session, hours=24, limit=100)
        kingfisher_rows = [sp for sp in recent if sp.scientific_name == "Alcedo atthis"]
        assert len(kingfisher_rows) == 1

    def test_ordered_most_recent_first(self, repo, session):
        s1 = _add_species(repo, session, "Garrulus glandarius", "Eurasian Jay")
        s2 = _add_species(repo, session, "Pica pica", "Eurasian Magpie")
        now = _now()
        repo.add_detection(session, s1.id, 0.7, now - timedelta(minutes=30))
        repo.add_detection(session, s2.id, 0.8, now - timedelta(minutes=5))
        session.commit()

        # Use a high limit so both species are guaranteed to appear
        recent = repo.get_recently_heard_species(session, hours=24, limit=100)
        ids = [sp.id for sp in recent]
        assert s2.id in ids, "Eurasian Magpie should be in results"
        assert s1.id in ids, "Eurasian Jay should be in results"
        assert ids.index(s2.id) < ids.index(s1.id), "Magpie (more recent) should come first"


# ---------------------------------------------------------------------------
# DetectionEvent tests
# ---------------------------------------------------------------------------

class TestDetectionEventCRUD:
    def test_add_event(self, repo, session):
        s = _add_species(repo, session, "Dendrocopos major", "Great Spotted Woodpecker")
        ev = repo.add_detection_event(
            session,
            species_id=s.id,
            started_at=_now(),
            peak_confidence=0.91,
            detection_count=4,
        )
        session.commit()
        assert ev.id is not None
        assert ev.detection_count == 4
        assert ev.ended_at is None

    def test_get_event_by_id(self, repo, session):
        s = _add_species(repo, session, "Sitta europaea", "Eurasian Nuthatch")
        ev = repo.add_detection_event(session, s.id, _now(), 0.75)
        session.commit()
        fetched = repo.get_detection_event_by_id(session, ev.id)
        assert fetched is not None
        assert fetched.peak_confidence == pytest.approx(0.75)

    def test_get_event_by_id_missing(self, repo, session):
        assert repo.get_detection_event_by_id(session, 999999) is None

    def test_list_events_newest_first(self, repo, session):
        s = _add_species(repo, session, "Troglodytes troglodytes", "Eurasian Wren")
        now = _now()
        repo.add_detection_event(session, s.id, now - timedelta(hours=2), 0.6)
        repo.add_detection_event(session, s.id, now - timedelta(hours=1), 0.7)
        repo.add_detection_event(session, s.id, now, 0.8)
        session.commit()

        events = repo.list_detection_events(session, species_id=s.id)
        starts = [e.started_at for e in events]
        assert starts == sorted(starts, reverse=True)


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_keys_present(self, repo, session):
        stats = repo.get_stats(session)
        assert "total_detections" in stats
        assert "total_species" in stats
        assert "total_events" in stats
        assert "latest_detection_at" in stats

    def test_stats_counts_increase(self, repo, session):
        before = repo.get_stats(session)

        s = _add_species(repo, session, "Regulus regulus", "Goldcrest")
        repo.add_detection(session, s.id, 0.88, _now())
        session.commit()

        after = repo.get_stats(session)
        assert after["total_detections"] > before["total_detections"]
        assert after["total_species"] > before["total_species"]

    def test_stats_latest_detection_none_when_empty(self, repo, session):
        # Fresh in-memory DB for this specific check
        fresh_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(fresh_engine)
        FreshSession = sessionmaker(bind=fresh_engine)
        fresh_session = FreshSession()

        stats = repo.get_stats(fresh_session)
        assert stats["latest_detection_at"] is None

        fresh_session.close()
        fresh_engine.dispose()
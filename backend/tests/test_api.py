"""
Tests for the BirdFrame FastAPI endpoints.

Uses FastAPI's TestClient (synchronous) with an in-memory SQLite
database seeded with known data so every assertion is deterministic.
"""

import sys
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database.models import Base
from backend.database.repository import DetectionRepository


# ---------------------------------------------------------------------------
# In-memory DB + app wiring
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mem_engine():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    engine = create_engine("sqlite://", creator=lambda: conn)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    conn.close()


@pytest.fixture(scope="module")
def seeded_session_factory(mem_engine):
    """Session factory pre-seeded with test data."""
    SessionLocal = sessionmaker(
        bind=mem_engine, autocommit=False, autoflush=False,
        expire_on_commit=False,
    )
    repo = DetectionRepository()
    session = SessionLocal()

    now = datetime.now(timezone.utc)

    # Seed species
    robin = repo.get_or_create_species(
        session, "Erithacus rubecula", "European Robin"
    )
    tit = repo.get_or_create_species(
        session, "Parus major", "Great Tit"
    )
    session.flush()

    # Seed detections
    repo.add_detection(session, robin.id, 0.9, now, 3.0, "Mock", "0.1")
    repo.add_detection(session, robin.id, 0.85, now - timedelta(minutes=5),
                       3.0, "Mock", "0.1")
    repo.add_detection(session, tit.id, 0.75, now - timedelta(hours=2),
                       3.0, "Mock", "0.1")
    session.commit()
    session.close()

    return SessionLocal


@pytest.fixture(scope="module")
def client(seeded_session_factory):
    """
    TestClient with the app's DB dependency overridden to use the
    in-memory seeded database.
    """
    from backend.api.main import app
    from backend.api.dependencies import get_db_session

    def override_get_db():
        session = seeded_session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health tests
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        data = response = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_timestamp(self, client):
        data = client.get("/health").json()
        assert "timestamp" in data

    def test_status_returns_200(self, client):
        response = client.get("/api/status")
        assert response.status_code == 200

    def test_status_has_config(self, client):
        data = client.get("/api/status").json()
        assert "config" in data
        assert "identifier_backend" in data["config"]

    def test_status_has_database(self, client):
        data = client.get("/api/status").json()
        assert "database" in data
        assert "total_detections" in data["database"]


# ---------------------------------------------------------------------------
# Detections tests
# ---------------------------------------------------------------------------

class TestDetections:
    def test_list_detections_200(self, client):
        response = client.get("/api/detections")
        assert response.status_code == 200

    def test_list_detections_structure(self, client):
        data = client.get("/api/detections").json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_list_detections_returns_seeded_data(self, client):
        data = client.get("/api/detections").json()
        assert data["total"] >= 3

    def test_list_detections_sorted_newest_first(self, client):
        data = client.get("/api/detections").json()
        items = data["items"]
        if len(items) > 1:
            timestamps = [i["timestamp"] for i in items]
            assert timestamps == sorted(timestamps, reverse=True)

    def test_list_detections_limit(self, client):
        data = client.get("/api/detections?limit=1").json()
        assert len(data["items"]) <= 1

    def test_list_detections_filter_by_species(self, client):
        # Get species id for robin
        species_data = client.get("/api/species").json()
        robin = next(
            s for s in species_data if s["scientific_name"] == "Erithacus rubecula"
        )
        data = client.get(f"/api/detections?species_id={robin['id']}").json()
        assert all(
            item["scientific_name"] == "Erithacus rubecula"
            for item in data["items"]
        )

    def test_get_detection_by_id(self, client):
        # Get first detection id
        items = client.get("/api/detections").json()["items"]
        first_id = items[0]["id"]
        response = client.get(f"/api/detections/{first_id}")
        assert response.status_code == 200
        assert response.json()["id"] == first_id

    def test_get_detection_not_found(self, client):
        response = client.get("/api/detections/999999")
        assert response.status_code == 404

    def test_detection_has_species_names(self, client):
        items = client.get("/api/detections").json()["items"]
        for item in items:
            assert "scientific_name" in item
            assert "common_name" in item
            assert len(item["scientific_name"]) > 0


# ---------------------------------------------------------------------------
# Species tests
# ---------------------------------------------------------------------------

class TestSpecies:
    def test_list_species_200(self, client):
        response = client.get("/api/species")
        assert response.status_code == 200

    def test_list_species_returns_seeded(self, client):
        data = client.get("/api/species").json()
        names = [s["scientific_name"] for s in data]
        assert "Erithacus rubecula" in names
        assert "Parus major" in names

    def test_list_species_alphabetical(self, client):
        data = client.get("/api/species").json()
        common_names = [s["common_name"] for s in data]
        assert common_names == sorted(common_names)

    def test_species_has_detection_count(self, client):
        data = client.get("/api/species").json()
        for s in data:
            assert "detection_count" in s
            assert s["detection_count"] >= 0

    def test_get_species_by_id(self, client):
        data = client.get("/api/species").json()
        first_id = data[0]["id"]
        response = client.get(f"/api/species/{first_id}")
        assert response.status_code == 200
        assert response.json()["id"] == first_id

    def test_get_species_not_found(self, client):
        response = client.get("/api/species/999999")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Heard recently tests
# ---------------------------------------------------------------------------

class TestHeardRecently:
    def test_heard_recently_200(self, client):
        response = client.get("/api/heard-recently")
        assert response.status_code == 200

    def test_heard_recently_structure(self, client):
        data = client.get("/api/heard-recently").json()
        assert "hours" in data
        assert "count" in data
        assert "species" in data

    def test_heard_recently_contains_recent_species(self, client):
        data = client.get("/api/heard-recently?hours=24").json()
        names = [s["scientific_name"] for s in data["species"]]
        # Robin was detected recently (just now in seed data)
        assert "Erithacus rubecula" in names

    def test_heard_recently_count_matches_species_list(self, client):
        data = client.get("/api/heard-recently").json()
        assert data["count"] == len(data["species"])

    def test_heard_recently_limit(self, client):
        data = client.get("/api/heard-recently?limit=1").json()
        assert len(data["species"]) <= 1


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_200(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200

    def test_stats_structure(self, client):
        data = client.get("/api/stats").json()
        assert "total_detections" in data
        assert "total_species" in data
        assert "total_events" in data
        assert "latest_detection_at" in data

    def test_stats_totals_match_seeded_data(self, client):
        data = client.get("/api/stats").json()
        assert data["total_detections"] >= 3
        assert data["total_species"] >= 2

    def test_stats_latest_detection_is_string(self, client):
        data = client.get("/api/stats").json()
        if data["latest_detection_at"] is not None:
            assert isinstance(data["latest_detection_at"], str)
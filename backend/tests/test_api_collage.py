"""
Tests for the collage API endpoints.

Uses the TestClient with the in-memory DB dependency override.
The collage generator is also overridden so tests never write
real files to disk or require artwork to be present.
"""

import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.api.main import app
from backend.api.dependencies import get_db_session
from backend.database.models import Base
from backend.database.repository import DetectionRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


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
    """Seed species + detections so heard-recently returns results."""
    SessionLocal = sessionmaker(
        bind=mem_engine, autocommit=False, autoflush=False,
        expire_on_commit=False,
    )
    repo = DetectionRepository()
    session = SessionLocal()
    now = datetime.now(timezone.utc)

    robin = repo.get_or_create_species(
        session, "Erithacus rubecula", "European Robin"
    )
    tit = repo.get_or_create_species(session, "Parus major", "Great Tit")
    session.flush()
    repo.add_detection(session, robin.id, 0.9, now, 3.0, "Mock", "0.1")
    repo.add_detection(session, tit.id, 0.8, now - timedelta(minutes=10),
                       3.0, "Mock", "0.1")
    session.commit()
    session.close()
    return SessionLocal


@pytest.fixture(scope="module")
def client(seeded_session_factory, tmp_path_factory):
    """TestClient with DB and collage dir overridden."""
    tmp_dir = tmp_path_factory.mktemp("collages")

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

    # Override the collage directory so tests write to tmp
    with patch("backend.api.routes.collage._generator") as mock_gen, \
         patch("backend.api.routes.collage._provider") as mock_prov, \
         patch("backend.api.routes.collage.config") as mock_config:

        mock_config.HEARD_RECENTLY_HOURS = 24
        mock_config.COLLAGE_MAX_SPECIES = 6
        mock_config.COLLAGE_DIR = tmp_dir

        # Provider: return a fake path for known species
        fake_artwork = tmp_dir / "fake_bird.jpg"
        from PIL import Image
        Image.new("RGB", (200, 300), (200, 180, 140)).save(str(fake_artwork))

        mock_prov.get_artwork.return_value = fake_artwork

        # Generator: write a real tiny JPEG to latest.jpg
        def fake_generate_latest(species_paths):
            out = tmp_dir / "latest.jpg"
            Image.new("RGB", (400, 300), (245, 240, 228)).save(str(out))
            return out

        mock_gen.generate_latest.side_effect = fake_generate_latest

        with TestClient(app) as c:
            yield c, tmp_dir

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Collage status tests
# ---------------------------------------------------------------------------

class TestCollageStatus:
    def test_status_when_no_collage(self):
        """Fresh state — no collage exists yet."""
        with patch(
            "backend.api.routes.collage._latest_path",
            return_value=Path("/nonexistent/latest.jpg"),
        ):
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                resp = c.get("/api/collage/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["exists"] is False
        assert data["path"] is None
        assert data["size_bytes"] is None

    def test_status_when_collage_exists(self, tmp_path):
        fake = tmp_path / "latest.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100)).save(str(fake))

        with patch(
            "backend.api.routes.collage._latest_path",
            return_value=fake,
        ):
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                resp = c.get("/api/collage/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["exists"] is True
        assert data["size_bytes"] > 0
        assert data["generated_at"] is not None
        assert data["age_seconds"] is not None
        assert data["age_seconds"] >= 0


# ---------------------------------------------------------------------------
# GET /api/collage/latest tests
# ---------------------------------------------------------------------------

class TestGetLatestCollage:
    def test_404_when_no_collage(self):
        with patch(
            "backend.api.routes.collage._latest_path",
            return_value=Path("/nonexistent/latest.jpg"),
        ):
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                resp = c.get("/api/collage/latest")
        assert resp.status_code == 404

    def test_returns_jpeg_when_exists(self, tmp_path):
        fake = tmp_path / "latest.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100)).save(str(fake))

        with patch(
            "backend.api.routes.collage._latest_path",
            return_value=fake,
        ):
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                resp = c.get("/api/collage/latest")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0


# ---------------------------------------------------------------------------
# POST /api/collage/generate tests
# ---------------------------------------------------------------------------

class TestGenerateCollage:
    def test_generate_returns_200(self, client):
        c, tmp_dir = client
        resp = c.post("/api/collage/generate")
        assert resp.status_code == 200

    def test_generate_response_structure(self, client):
        c, tmp_dir = client
        data = c.post("/api/collage/generate").json()
        assert "success" in data
        assert "species_count" in data
        assert "message" in data
        assert "generated_at" in data

    def test_generate_success_true_when_species_available(self, client):
        c, tmp_dir = client
        data = c.post("/api/collage/generate").json()
        assert data["success"] is True

    def test_generate_species_count_positive(self, client):
        c, tmp_dir = client
        data = c.post("/api/collage/generate").json()
        assert data["species_count"] >= 0

    def test_generate_with_hours_param(self, client):
        c, tmp_dir = client
        resp = c.post("/api/collage/generate?hours=48")
        assert resp.status_code == 200

    def test_generate_with_limit_param(self, client):
        c, tmp_dir = client
        resp = c.post("/api/collage/generate?limit=3")
        assert resp.status_code == 200

    def test_generate_invalid_hours_rejected(self, client):
        c, tmp_dir = client
        resp = c.post("/api/collage/generate?hours=0")
        assert resp.status_code == 422  # FastAPI validation error

    def test_generate_invalid_limit_rejected(self, client):
        c, tmp_dir = client
        resp = c.post("/api/collage/generate?limit=0")
        assert resp.status_code == 422

    def test_generate_no_species_returns_success_false(self, tmp_path):
        """When no species detected, success should be False."""
        import sqlite3
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # Empty database — no detections
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        engine = create_engine("sqlite://", creator=lambda: conn)
        Base.metadata.create_all(engine)
        EmptySession = sessionmaker(bind=engine, autocommit=False,
                                    autoflush=False)

        def empty_db():
            s = EmptySession()
            try:
                yield s
            finally:
                s.close()

        app.dependency_overrides[get_db_session] = empty_db
        with TestClient(app) as c:
            resp = c.post("/api/collage/generate")
        app.dependency_overrides.clear()
        conn.close()

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False


# ---------------------------------------------------------------------------
# Background generation test
# ---------------------------------------------------------------------------

class TestGenerateBackground:
    def test_background_returns_accepted(self, client):
        c, tmp_dir = client
        resp = c.post("/api/collage/generate-background")
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert "message" in data
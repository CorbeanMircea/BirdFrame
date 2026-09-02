"""
Tests for DetectionService.

All tests use:
  - MockBirdIdentifier  (no real model)
  - In-memory SQLite    (no real database file)
  - Synthetic numpy audio (no real microphone)

SQLite in-memory isolation note
--------------------------------
SQLite :memory: databases are connection-scoped: each new connection
gets an empty database. To share schema + data across the service's
background thread and the test's query session we use a single
persistent connection for the entire test, passed via
creator=lambda: shared_conn to create_engine.
"""

import sys
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.detection.service import DetectionService, DetectionServiceError
from backend.identification.mock_identifier import MockBirdIdentifier
from backend.identification.base import BirdIdentifierError
from backend.audio.detector import BirdDetector
from backend.audio.processor import AudioProcessor
from backend.database.models import Base, Detection, Species
from backend.database.repository import DetectionRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


@pytest.fixture
def mem_engine():
    """
    In-memory SQLite engine backed by a single shared connection.

    Using creator=lambda: conn forces SQLAlchemy to reuse the same
    underlying sqlite3 connection for every Session it opens, so the
    schema created by Base.metadata.create_all() is visible to all
    threads and sessions throughout the test.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    engine = create_engine("sqlite://", creator=lambda: conn)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    conn.close()


@pytest.fixture
def session_factory(mem_engine):
    """Session factory bound to the shared-connection engine."""
    return sessionmaker(
        bind=mem_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


@pytest.fixture
def query_session(session_factory):
    """Session for querying results after the service has run."""
    session = session_factory()
    yield session
    session.close()


def _sine(seconds=3.0, freq=4000.0, sr=SAMPLE_RATE, amplitude=0.5):
    """Bird-frequency sine wave that passes the detector."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds=3.0, sr=SAMPLE_RATE):
    return np.zeros(int(seconds * sr), dtype=np.float32)


def _make_detector():
    """Detector tuned for 16 kHz test audio (freq_max below Nyquist)."""
    return BirdDetector(
        energy_threshold=0.001,
        freq_min=1000,
        freq_max=7000,
        band_ratio_threshold=0.05,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        identifier = MockBirdIdentifier(mode="fixed")
        service = DetectionService(identifier=identifier)
        assert not service.is_running

    def test_repr(self):
        service = DetectionService(identifier=MockBirdIdentifier())
        assert "DetectionService" in repr(service)

    def test_custom_detector_accepted(self):
        service = DetectionService(
            identifier=MockBirdIdentifier(mode="fixed"),
            detector=BirdDetector(energy_threshold=0.0, band_ratio_threshold=0.0),
        )
        assert service is not None

    def test_custom_processor_accepted(self):
        segments = []
        processor = AudioProcessor(
            segment_callback=lambda s, r: segments.append(s),
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        service = DetectionService(
            identifier=MockBirdIdentifier(mode="fixed"),
            processor=processor,
        )
        assert service is not None

    def test_custom_session_factory_accepted(self, session_factory):
        service = DetectionService(
            identifier=MockBirdIdentifier(mode="fixed"),
            session_factory=session_factory,
        )
        assert service is not None


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_sets_running(self):
        service = DetectionService(identifier=MockBirdIdentifier(mode="fixed"))
        service.start()
        assert service.is_running is True
        service.stop()

    def test_stop_clears_running(self):
        service = DetectionService(identifier=MockBirdIdentifier(mode="fixed"))
        service.start()
        service.stop()
        assert service.is_running is False

    def test_double_start_safe(self):
        service = DetectionService(identifier=MockBirdIdentifier(mode="fixed"))
        service.start()
        service.start()  # must not raise
        assert service.is_running is True
        service.stop()

    def test_stop_without_start_safe(self):
        service = DetectionService(identifier=MockBirdIdentifier(mode="fixed"))
        service.stop()  # must not raise

    def test_handle_chunk_ignored_when_not_running(self):
        service = DetectionService(identifier=MockBirdIdentifier(mode="fixed"))
        service.handle_chunk(_sine(), SAMPLE_RATE)
        assert service.get_stats()["chunks_received"] == 0


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_keys_present(self):
        service = DetectionService(identifier=MockBirdIdentifier())
        stats = service.get_stats()
        assert "running" in stats
        assert "chunks_received" in stats
        assert "segments_analysed" in stats
        assert "segments_accepted" in stats
        assert "detections_saved" in stats
        assert "identifier" in stats
        assert "identifier_version" in stats

    def test_stats_identifier_name(self):
        service = DetectionService(identifier=MockBirdIdentifier())
        assert service.get_stats()["identifier"] == "MockBirdIdentifier"

    def test_chunks_received_increments(self):
        service = DetectionService(
            identifier=MockBirdIdentifier(mode="empty"),
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        service.start()
        service.handle_chunk(_sine(1.0), SAMPLE_RATE)
        service.stop()
        assert service.get_stats()["chunks_received"] == 1


# ---------------------------------------------------------------------------
# Pipeline flow tests (no DB writes)
# ---------------------------------------------------------------------------

class TestPipelineFlow:
    def test_silence_not_accepted_by_detector(self):
        identifier = MockBirdIdentifier(mode="fixed")
        service = DetectionService(
            identifier=identifier,
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        service.start()
        for _ in range(3):
            service.handle_chunk(_silence(1.0), SAMPLE_RATE)
        service.stop()

        assert service.get_stats()["segments_accepted"] == 0
        assert identifier.call_count == 0

    def test_bird_audio_reaches_identifier(self):
        identifier = MockBirdIdentifier(mode="empty")
        service = DetectionService(
            identifier=identifier,
            detector=_make_detector(),
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        service.start()
        service.handle_chunk(_sine(1.0), SAMPLE_RATE)
        service.stop()

        assert service.get_stats()["segments_accepted"] >= 1
        assert identifier.call_count >= 1

    def test_low_confidence_not_saved(self):
        identifier = MockBirdIdentifier(mode="fixed", fixed_confidence=0.3)
        service = DetectionService(
            identifier=identifier,
            detector=BirdDetector(energy_threshold=0.0, band_ratio_threshold=0.0),
            min_confidence=0.8,
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        service.start()
        service.handle_chunk(_sine(1.0), SAMPLE_RATE)
        service.stop()

        assert service.get_stats()["detections_saved"] == 0

    def test_empty_identifier_saves_nothing(self):
        service = DetectionService(
            identifier=MockBirdIdentifier(mode="empty"),
            detector=BirdDetector(energy_threshold=0.0, band_ratio_threshold=0.0),
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        service.start()
        service.handle_chunk(_sine(1.0), SAMPLE_RATE)
        service.stop()

        assert service.get_stats()["detections_saved"] == 0


# ---------------------------------------------------------------------------
# End-to-end integration tests (real in-memory DB writes)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_detections_written_to_database(self, session_factory, query_session):
        """Full pipeline: audio → detector → identifier → DB."""
        service = DetectionService(
            identifier=MockBirdIdentifier(mode="fixed", fixed_confidence=0.9),
            detector=_make_detector(),
            session_factory=session_factory,
            min_confidence=0.5,
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        service.start()
        service.handle_chunk(_sine(1.0), SAMPLE_RATE)
        service.stop()

        detections = query_session.query(Detection).all()
        species = query_session.query(Species).all()

        assert len(detections) > 0, "At least one detection should be saved"
        assert len(species) > 0, "At least one species should be created"
        assert all(d.confidence >= 0.5 for d in detections)

    def test_species_created_once_for_repeated_detections(
        self, session_factory, query_session
    ):
        """Detecting the same species twice must create only one Species row."""
        service = DetectionService(
            identifier=MockBirdIdentifier(mode="fixed", fixed_confidence=0.9),
            detector=_make_detector(),
            session_factory=session_factory,
            min_confidence=0.5,
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        service.start()
        service.handle_chunk(_sine(1.0), SAMPLE_RATE)
        service.handle_chunk(_sine(1.0), SAMPLE_RATE)
        service.stop()

        robin_count = (
            query_session.query(Species)
            .filter(Species.scientific_name == "Erithacus rubecula")
            .count()
        )
        assert robin_count == 1, "Species must not be duplicated"

    def test_mock_recorder_to_service_pipeline(self, session_factory, query_session):
        """
        Full end-to-end: MockAudioRecorder → DetectionService → DB.

        The shared sqlite3 connection (via mem_engine fixture) ensures
        the schema and all writes are visible across threads.
        """
        from backend.audio.mock_recorder import MockAudioRecorder

        service = DetectionService(
            identifier=MockBirdIdentifier(mode="fixed", fixed_confidence=0.9),
            detector=_make_detector(),
            session_factory=session_factory,
            min_confidence=0.5,
            segment_duration=1.0,
            overlap_duration=0.0,
        )

        service.start()
        recorder = MockAudioRecorder.from_array(
            _sine(seconds=3.0),
            sample_rate=SAMPLE_RATE,
            callback=service.handle_chunk,
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=10.0)
        service.stop()

        detections = query_session.query(Detection).all()

        assert len(detections) > 0, "At least one detection should be saved"
        stats = service.get_stats()
        assert stats["chunks_received"] == 3
        assert stats["detections_saved"] > 0
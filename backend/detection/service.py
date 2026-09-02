"""
DetectionService — orchestrates the full bird detection pipeline.

Wires together:
    AudioProcessor  → BirdDetector → BirdIdentifier → DetectionRepository

The service owns the pipeline lifecycle (start / stop) and handles
all coordination between components. Individual components remain
unaware of each other.

Data flow:

    AudioRecorder (external)
        └─► DetectionService.handle_chunk(chunk, sample_rate)
                └─► AudioProcessor
                        └─► [segment ready]
                                └─► BirdDetector.is_bird_audio()
                                        └─► [if accepted]
                                                └─► BirdIdentifier.identify()
                                                        └─► [filter by confidence]
                                                                └─► DetectionRepository
                                                                        └─► SQLite

Usage:

    from backend.detection.service import DetectionService
    from backend.identification.mock_identifier import MockBirdIdentifier
    from backend.audio.recorder import AudioRecorder

    service = DetectionService(identifier=MockBirdIdentifier())
    service.start()

    recorder = AudioRecorder(callback=service.handle_chunk)
    recorder.start()
    ...
    recorder.stop()
    service.stop()
"""

import sys
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.audio.processor import AudioProcessor
from backend.audio.detector import BirdDetector
from backend.identification.base import BirdIdentifier, IdentificationResult
from backend.database.engine import get_session, init_db
from backend.database.repository import DetectionRepository

logger = logging.getLogger(__name__)

# Type alias for session factory callables
SessionFactory = Callable[[], object]


class DetectionServiceError(Exception):
    """Raised when DetectionService encounters an unrecoverable error."""
    pass


class DetectionService:
    """
    Orchestrates the full bird detection pipeline.

    Parameters
    ----------
    identifier : BirdIdentifier
        The identification backend to use. Must implement identify().
    detector : BirdDetector | None
        Pre-filter. If None, a default BirdDetector is created from
        config values.
    processor : AudioProcessor | None
        Audio chunker. If None, one is created from config values,
        wired to feed into the detector → identifier chain.
    repository : DetectionRepository | None
        Database access. If None, a default DetectionRepository is used.
    session_factory : callable | None
        A zero-argument callable that returns a SQLAlchemy Session.
        Defaults to the standard get_session() from database.engine.
        Inject a custom factory in tests to use an in-memory database
        without patching — this works correctly across threads.
    min_confidence : float
        Minimum confidence for a result to be persisted.
        Defaults to config.IDENTIFIER_MIN_CONFIDENCE.
    segment_duration : float
        Segment length passed to AudioProcessor if one is auto-created.
    overlap_duration : float
        Overlap passed to AudioProcessor if one is auto-created.
    target_sample_rate : int | None
        Resample target passed to AudioProcessor. Useful when recorder
        runs at 44100 but the identifier expects 48000.
    """

    def __init__(
        self,
        identifier: BirdIdentifier,
        detector: Optional[BirdDetector] = None,
        processor: Optional[AudioProcessor] = None,
        repository: Optional[DetectionRepository] = None,
        session_factory: Optional[SessionFactory] = None,
        min_confidence: float = config.IDENTIFIER_MIN_CONFIDENCE,
        segment_duration: float = config.AUDIO_CHUNK_DURATION,
        overlap_duration: float = config.AUDIO_CHUNK_OVERLAP,
        target_sample_rate: Optional[int] = None,
    ) -> None:
        self._identifier = identifier
        self._detector = detector or BirdDetector()
        self._repository = repository or DetectionRepository()
        self._session_factory = session_factory or get_session
        self._min_confidence = min_confidence
        self._lock = threading.Lock()
        self._running = False

        # Stats counters
        self._chunks_received: int = 0
        self._segments_analysed: int = 0
        self._segments_accepted: int = 0
        self._detections_saved: int = 0

        # Build or accept the AudioProcessor, wiring its output to
        # our internal _on_segment handler.
        if processor is not None:
            self._processor = processor
        else:
            self._processor = AudioProcessor(
                segment_callback=self._on_segment,
                segment_duration=segment_duration,
                overlap_duration=overlap_duration,
                target_sample_rate=target_sample_rate,
            )

        logger.debug(
            "DetectionService created: identifier=%r min_confidence=%.2f",
            identifier.model_name,
            min_confidence,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """
        Prepare the service for operation.

        - Ensures the database schema exists (skipped if a custom
          session_factory was injected, since the test controls the schema).
        - Calls identifier.warmup() so model weights are loaded before
          the first real audio arrives.
        """
        with self._lock:
            if self._running:
                logger.warning("DetectionService.start() called while already running.")
                return

            logger.info("DetectionService starting…")

            # Only run init_db when using the default session factory.
            # Tests that inject their own factory manage their own schema.
            if self._session_factory is get_session:
                init_db()

            try:
                self._identifier.warmup()
            except Exception as exc:
                raise DetectionServiceError(
                    f"Identifier warmup failed: {exc}"
                ) from exc

            self._running = True
            logger.info(
                "DetectionService started (identifier=%s %s).",
                self._identifier.model_name,
                self._identifier.model_version,
            )

    def stop(self) -> None:
        """Stop the service and reset the audio processor buffer."""
        with self._lock:
            if not self._running:
                return
            self._processor.reset()
            self._running = False
            logger.info(
                "DetectionService stopped. "
                "chunks=%d segments_analysed=%d accepted=%d saved=%d",
                self._chunks_received,
                self._segments_analysed,
                self._segments_accepted,
                self._detections_saved,
            )

    # ------------------------------------------------------------------
    # Audio entry point
    # ------------------------------------------------------------------

    def handle_chunk(self, chunk: np.ndarray, sample_rate: int) -> None:
        """
        Accept a raw audio chunk from an AudioRecorder.

        This is the callback to pass to AudioRecorder:

            recorder = AudioRecorder(callback=service.handle_chunk)

        Thread-safe: sounddevice calls this on its own thread.
        """
        if not self._running:
            return

        self._chunks_received += 1
        try:
            self._processor.process(chunk, sample_rate)
        except Exception as exc:
            logger.error("AudioProcessor error in handle_chunk: %s", exc)

    # ------------------------------------------------------------------
    # Internal pipeline stages
    # ------------------------------------------------------------------

    def _on_segment(self, segment: np.ndarray, sample_rate: int) -> None:
        """
        Called by AudioProcessor when a complete segment is ready.

        Runs: detector → identifier → repository.
        """
        self._segments_analysed += 1

        # Stage 1: pre-filter
        try:
            result = self._detector.analyse(segment, sample_rate)
        except Exception as exc:
            logger.error("BirdDetector error: %s", exc)
            return

        if not result.accepted:
            logger.debug(
                "Segment rejected by detector: %s", result.rejection_reason
            )
            return

        self._segments_accepted += 1
        logger.debug(
            "Segment accepted (rms=%.4f band_ratio=%.3f) — running identifier.",
            result.rms_energy,
            result.band_energy_ratio,
        )

        # Stage 2: identify
        segment_start = datetime.now(timezone.utc)
        try:
            candidates = self._identifier.identify_and_filter(
                segment,
                sample_rate,
                min_confidence=self._min_confidence,
            )
        except Exception as exc:
            logger.error("BirdIdentifier error: %s", exc)
            return

        if not candidates:
            logger.debug("Identifier returned no results above threshold.")
            return

        # Stage 3: persist
        self._persist_results(candidates, segment_start, len(segment) / sample_rate)

    def _persist_results(
        self,
        candidates: list[IdentificationResult],
        timestamp: datetime,
        duration_seconds: float,
    ) -> None:
        """Write accepted identification results to the database."""
        try:
            with self._session_factory() as session:
                for result in candidates:
                    species = self._repository.get_or_create_species(
                        session,
                        scientific_name=result.scientific_name,
                        common_name=result.common_name,
                    )
                    self._repository.add_detection(
                        session,
                        species_id=species.id,
                        confidence=result.confidence,
                        timestamp=timestamp,
                        duration_seconds=duration_seconds,
                        model_name=result.model_name,
                        model_version=result.model_version,
                    )
                    self._detections_saved += 1
                    logger.info(
                        "Detection saved: %s (confidence=%.2f)",
                        result.scientific_name,
                        result.confidence,
                    )
                session.commit()
        except Exception as exc:
            logger.error("Database error in _persist_results: %s", exc)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return pipeline counters for monitoring / API."""
        return {
            "running": self._running,
            "chunks_received": self._chunks_received,
            "segments_analysed": self._segments_analysed,
            "segments_accepted": self._segments_accepted,
            "detections_saved": self._detections_saved,
            "identifier": self._identifier.model_name,
            "identifier_version": self._identifier.model_version,
        }

    def __repr__(self) -> str:
        return (
            f"<DetectionService running={self._running} "
            f"identifier={self._identifier.model_name!r}>"
        )
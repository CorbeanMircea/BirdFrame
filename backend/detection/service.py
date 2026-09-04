"""
DetectionService — orchestrates the full bird detection pipeline.

Wires together:
    AudioProcessor → BirdDetector → BirdIdentifier
    → DetectionRepository → DetectionGrouper

Data flow:

    AudioRecorder (external)
        └─► DetectionService.handle_chunk(chunk, sample_rate)
                └─► AudioProcessor
                        └─► [segment ready]
                                └─► BirdDetector.is_bird_audio()
                                        └─► [if accepted]
                                                └─► BirdIdentifier.identify()
                                                        └─► [filter by confidence]
                                                                └─► DetectionRepository (save Detection)
                                                                        └─► DetectionGrouper (group into Event)

Usage:

    from backend.detection.service import DetectionService
    from backend.identification.mock_identifier import MockBirdIdentifier

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
from backend.detection.grouper import DetectionGrouper

logger = logging.getLogger(__name__)

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
        The identification backend to use.
    detector : BirdDetector | None
        Pre-filter. Defaults to a BirdDetector from config values.
    processor : AudioProcessor | None
        Audio chunker. Defaults to one built from config values.
    repository : DetectionRepository | None
        Database access. Defaults to a new DetectionRepository.
    grouper : DetectionGrouper | None
        Groups consecutive detections into events. Defaults to a new
        DetectionGrouper. Pass None to disable grouping entirely.
    session_factory : callable | None
        Zero-argument callable returning a SQLAlchemy Session.
        Defaults to get_session(). Inject a custom factory in tests.
    min_confidence : float
        Minimum confidence for a result to be persisted.
    segment_duration : float
        Segment length for the auto-created AudioProcessor.
    overlap_duration : float
        Overlap for the auto-created AudioProcessor.
    target_sample_rate : int | None
        Resample target for the auto-created AudioProcessor.
    """

    def __init__(
        self,
        identifier: BirdIdentifier,
        detector: Optional[BirdDetector] = None,
        processor: Optional[AudioProcessor] = None,
        repository: Optional[DetectionRepository] = None,
        grouper: Optional[DetectionGrouper] = None,
        session_factory: Optional[SessionFactory] = None,
        min_confidence: float = config.IDENTIFIER_MIN_CONFIDENCE,
        segment_duration: float = config.AUDIO_CHUNK_DURATION,
        overlap_duration: float = config.AUDIO_CHUNK_OVERLAP,
        target_sample_rate: Optional[int] = None,
        enable_grouping: bool = True,
    ) -> None:
        self._identifier = identifier
        self._detector = detector or BirdDetector()
        self._repository = repository or DetectionRepository()
        self._session_factory = session_factory or get_session
        self._min_confidence = min_confidence
        self._lock = threading.Lock()
        self._running = False
        self._enable_grouping = enable_grouping

        # Grouper: use provided, or create default, or disable
        if grouper is not None:
            self._grouper: Optional[DetectionGrouper] = grouper
        elif enable_grouping:
            self._grouper = DetectionGrouper(self._repository)
        else:
            self._grouper = None

        # Stats counters
        self._chunks_received: int = 0
        self._segments_analysed: int = 0
        self._segments_accepted: int = 0
        self._detections_saved: int = 0
        self._events_created_or_extended: int = 0

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
            "DetectionService created: identifier=%r min_confidence=%.2f "
            "grouping=%s",
            identifier.model_name, min_confidence, enable_grouping,
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

        Ensures the DB schema exists (when using the default session
        factory) and warms up the identifier.
        """
        with self._lock:
            if self._running:
                logger.warning("DetectionService.start() called while already running.")
                return

            logger.info("DetectionService starting…")

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
                "DetectionService started (identifier=%s %s grouping=%s).",
                self._identifier.model_name,
                self._identifier.model_version,
                self._enable_grouping,
            )

    def stop(self) -> None:
        """Stop the service, close stale events, and reset the processor."""
        with self._lock:
            if not self._running:
                return

            # Close any events that are still open
            if self._grouper is not None:
                try:
                    with self._session_factory() as session:
                        closed = self._grouper.close_stale_events(session)
                        session.commit()
                        if closed:
                            logger.info(
                                "Closed %d stale event(s) on stop.", closed
                            )
                except Exception as exc:
                    logger.error("Error closing stale events on stop: %s", exc)

            self._processor.reset()
            self._running = False
            logger.info(
                "DetectionService stopped. "
                "chunks=%d analysed=%d accepted=%d saved=%d events=%d",
                self._chunks_received,
                self._segments_analysed,
                self._segments_accepted,
                self._detections_saved,
                self._events_created_or_extended,
            )

    # ------------------------------------------------------------------
    # Audio entry point
    # ------------------------------------------------------------------

    def handle_chunk(self, chunk: np.ndarray, sample_rate: int) -> None:
        """
        Accept a raw audio chunk from AudioRecorder.

        Pass this as the callback:
            recorder = AudioRecorder(callback=service.handle_chunk)
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
        """Called by AudioProcessor when a complete segment is ready."""
        self._segments_analysed += 1

        # Stage 1: pre-filter
        try:
            det_result = self._detector.analyse(segment, sample_rate)
        except Exception as exc:
            logger.error("BirdDetector error: %s", exc)
            return

        if not det_result.accepted:
            logger.debug("Segment rejected: %s", det_result.rejection_reason)
            return

        self._segments_accepted += 1

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

        # Stage 3: persist + group
        self._persist_and_group(
            candidates, segment_start, len(segment) / sample_rate
        )

    def _persist_and_group(
        self,
        candidates: list[IdentificationResult],
        timestamp: datetime,
        duration_seconds: float,
    ) -> None:
        """Save detections to the DB and run the grouper on each."""
        try:
            with self._session_factory() as session:
                for result in candidates:
                    # Save species
                    species = self._repository.get_or_create_species(
                        session,
                        scientific_name=result.scientific_name,
                        common_name=result.common_name,
                    )

                    # Save detection
                    detection = self._repository.add_detection(
                        session,
                        species_id=species.id,
                        confidence=result.confidence,
                        timestamp=timestamp,
                        duration_seconds=duration_seconds,
                        model_name=result.model_name,
                        model_version=result.model_version,
                    )
                    self._detections_saved += 1

                    # Group into event
                    if self._grouper is not None:
                        try:
                            self._grouper.process(session, detection)
                            self._events_created_or_extended += 1
                        except Exception as exc:
                            logger.error(
                                "DetectionGrouper error for %s: %s",
                                result.scientific_name, exc,
                            )

                    logger.info(
                        "Saved: %s confidence=%.2f",
                        result.scientific_name, result.confidence,
                    )

                session.commit()

        except Exception as exc:
            logger.error("Database error in _persist_and_group: %s", exc)

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
            "events_created_or_extended": self._events_created_or_extended,
            "grouping_enabled": self._enable_grouping,
            "identifier": self._identifier.model_name,
            "identifier_version": self._identifier.model_version,
        }

    def __repr__(self) -> str:
        return (
            f"<DetectionService running={self._running} "
            f"identifier={self._identifier.model_name!r} "
            f"grouping={self._enable_grouping}>"
        )
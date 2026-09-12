"""
DetectionService — orchestrates the full bird detection pipeline.

Wires together:
    AudioProcessor → BirdDetector → BirdIdentifier
    → DetectionRepository → DetectionGrouper → ArtworkTrigger

Data flow:

    AudioRecorder (external)
        └─► DetectionService.handle_chunk(chunk, sample_rate)
                └─► AudioProcessor
                        └─► [segment ready]
                                └─► BirdDetector
                                        └─► [if accepted]
                                                └─► BirdIdentifier
                                                        └─► [filter by confidence]
                                                                └─► DetectionRepository
                                                                        └─► DetectionGrouper
                                                                                └─► ArtworkTrigger
                                                                                    (background)
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
    pass


class DetectionService:
    """
    Orchestrates the full bird detection pipeline.

    Parameters
    ----------
    identifier : BirdIdentifier
    detector : BirdDetector | None
    processor : AudioProcessor | None
    repository : DetectionRepository | None
    grouper : DetectionGrouper | None
    artwork_trigger : ArtworkTrigger | None
        If provided, called after each detection to trigger background
        artwork generation for new species.
    session_factory : callable | None
    min_confidence : float
    segment_duration : float
    overlap_duration : float
    target_sample_rate : int | None
    enable_grouping : bool
    """

    def __init__(
        self,
        identifier: BirdIdentifier,
        detector: Optional[BirdDetector] = None,
        processor: Optional[AudioProcessor] = None,
        repository: Optional[DetectionRepository] = None,
        grouper: Optional[DetectionGrouper] = None,
        artwork_trigger=None,
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
        self._artwork_trigger = artwork_trigger
        self._min_confidence = min_confidence
        self._lock = threading.Lock()
        self._running = False
        self._enable_grouping = enable_grouping

        if grouper is not None:
            self._grouper: Optional[DetectionGrouper] = grouper
        elif enable_grouping:
            self._grouper = DetectionGrouper(self._repository)
        else:
            self._grouper = None

        # Stats
        self._chunks_received: int = 0
        self._segments_analysed: int = 0
        self._segments_accepted: int = 0
        self._detections_saved: int = 0
        self._events_created_or_extended: int = 0
        self._artwork_triggered: int = 0

        if processor is not None:
            self._processor = processor
        else:
            self._processor = AudioProcessor(
                segment_callback=self._on_segment,
                segment_duration=segment_duration,
                overlap_duration=overlap_duration,
                target_sample_rate=target_sample_rate,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
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

            # Start artwork trigger if provided
            if self._artwork_trigger is not None:
                self._artwork_trigger.start()
                if self._artwork_trigger.is_running:
                    logger.info(
                        "ArtworkTrigger active — new species will auto-generate artwork."
                    )
                else:
                    logger.info(
                        "ArtworkTrigger inactive (ComfyUI unavailable or "
                        "static backend)."
                    )

            self._running = True
            logger.info(
                "DetectionService started (identifier=%s %s grouping=%s artwork_trigger=%s).",
                self._identifier.model_name,
                self._identifier.model_version,
                self._enable_grouping,
                self._artwork_trigger is not None,
            )

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return

            if self._grouper is not None:
                try:
                    with self._session_factory() as session:
                        closed = self._grouper.close_stale_events(session)
                        session.commit()
                        if closed:
                            logger.info("Closed %d stale event(s).", closed)
                except Exception as exc:
                    logger.error("Error closing stale events: %s", exc)

            if self._artwork_trigger is not None:
                self._artwork_trigger.stop()

            self._processor.reset()
            self._running = False
            logger.info(
                "DetectionService stopped. "
                "chunks=%d analysed=%d accepted=%d saved=%d "
                "events=%d artwork_triggered=%d",
                self._chunks_received,
                self._segments_analysed,
                self._segments_accepted,
                self._detections_saved,
                self._events_created_or_extended,
                self._artwork_triggered,
            )

    # ------------------------------------------------------------------
    # Audio entry point
    # ------------------------------------------------------------------

    def handle_chunk(self, chunk: np.ndarray, sample_rate: int) -> None:
        if not self._running:
            return
        self._chunks_received += 1
        try:
            self._processor.process(chunk, sample_rate)
        except Exception as exc:
            logger.error("AudioProcessor error: %s", exc)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _on_segment(self, segment: np.ndarray, sample_rate: int) -> None:
        self._segments_analysed += 1

        try:
            det_result = self._detector.analyse(segment, sample_rate)
        except Exception as exc:
            logger.error("BirdDetector error: %s", exc)
            return

        if not det_result.accepted:
            return

        self._segments_accepted += 1

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
            return

        self._persist_and_group(
            candidates, segment_start, len(segment) / sample_rate
        )

    def _persist_and_group(
        self,
        candidates: list[IdentificationResult],
        timestamp: datetime,
        duration_seconds: float,
    ) -> None:
        try:
            with self._session_factory() as session:
                for result in candidates:
                    species = self._repository.get_or_create_species(
                        session,
                        scientific_name=result.scientific_name,
                        common_name=result.common_name,
                    )

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

                    if self._grouper is not None:
                        try:
                            self._grouper.process(session, detection)
                            self._events_created_or_extended += 1
                        except Exception as exc:
                            logger.error("DetectionGrouper error: %s", exc)

                    # Trigger artwork generation for new species
                    if self._artwork_trigger is not None:
                        try:
                            self._artwork_trigger.on_detection(
                                result.scientific_name,
                                result.common_name,
                            )
                            self._artwork_triggered += 1
                        except Exception as exc:
                            logger.error("ArtworkTrigger error: %s", exc)

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
        artwork_stats = {}
        if self._artwork_trigger is not None:
            artwork_stats = {
                "artwork_trigger_active": self._artwork_trigger.is_running,
                "artwork_in_progress": list(
                    self._artwork_trigger.species_in_progress
                ),
                "artwork_completed": list(
                    self._artwork_trigger.species_completed
                ),
            }

        return {
            "running": self._running,
            "chunks_received": self._chunks_received,
            "segments_analysed": self._segments_analysed,
            "segments_accepted": self._segments_accepted,
            "detections_saved": self._detections_saved,
            "events_created_or_extended": self._events_created_or_extended,
            "artwork_triggered": self._artwork_triggered,
            "grouping_enabled": self._enable_grouping,
            "identifier": self._identifier.model_name,
            "identifier_version": self._identifier.model_version,
            **artwork_stats,
        }

    def __repr__(self) -> str:
        return (
            f"<DetectionService running={self._running} "
            f"identifier={self._identifier.model_name!r} "
            f"grouping={self._enable_grouping} "
            f"artwork_trigger={self._artwork_trigger is not None}>"
        )
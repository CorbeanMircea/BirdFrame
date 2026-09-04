"""
DetectionGrouper — merges nearby same-species detections into events.

Problem it solves:
    A bird singing for 30 seconds will produce ~10 consecutive 3-second
    detections. Without grouping, the database has 10 rows and the UI
    shows "Robin detected 10 times". With grouping, those 10 detections
    collapse into one DetectionEvent: "Robin present for ~30 seconds".

Algorithm:
    For each new Detection:
    1. Find the most recent DetectionEvent for the same species where
       the event is still "active" — meaning its ended_at (last
       detection time) is within gap_seconds of this detection.
    2. If found: extend the event (update ended_at, peak_confidence,
       detection_count).
    3. Otherwise: create a new event.

    ended_at semantics in this module:
        None      → event has exactly one detection (just started)
        timestamp → timestamp of the most recent detection in this event

    An event is "open/active" if:
        ended_at is None  (just one detection, started_at is recent)
        OR
        ended_at is recent (within gap_seconds of the new detection)

    An event is "closed/finished" when close_stale_events() is called
    explicitly — we do not auto-close events during normal processing.

Timezone note:
    SQLite stores datetimes without timezone info. SQLAlchemy reads them
    back as timezone-naive datetime objects. _as_naive_utc() normalises
    both aware and naive datetimes to naive UTC before arithmetic.
"""

import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.database.models import Detection, DetectionEvent
from backend.database.repository import DetectionRepository

logger = logging.getLogger(__name__)


def _as_naive_utc(dt: datetime) -> datetime:
    """Normalise a datetime to timezone-naive UTC."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class DetectionGrouper:
    """
    Merges consecutive same-species detections into DetectionEvents.

    Parameters
    ----------
    repository : DetectionRepository
    gap_seconds : float
        Maximum seconds between consecutive detections for them to be
        part of the same event. Defaults to config.GROUPING_GAP_SECONDS.
    """

    def __init__(
        self,
        repository: DetectionRepository,
        gap_seconds: float = config.GROUPING_GAP_SECONDS,
    ) -> None:
        if gap_seconds <= 0:
            raise ValueError(
                f"gap_seconds must be positive, got {gap_seconds}."
            )
        self._repo = repository
        self._gap_seconds = gap_seconds
        logger.debug("DetectionGrouper initialised: gap_seconds=%.1f", gap_seconds)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def gap_seconds(self) -> float:
        return self._gap_seconds

    def process(
        self,
        session,
        detection: Detection,
    ) -> DetectionEvent:
        """
        Group *detection* into an existing or new DetectionEvent.

        The detection must already be flushed (detection.id is set).
        The caller must call session.commit() to make writes durable.
        """
        if detection.id is None:
            raise ValueError(
                "detection must be flushed before calling process() "
                "(detection.id is None)."
            )

        existing_event = self._find_active_event(
            session, detection.species_id, detection.timestamp
        )

        if existing_event is not None:
            event = self._extend_event(session, existing_event, detection)
            logger.debug(
                "Extended event id=%d species_id=%d count=%d peak=%.2f",
                event.id, detection.species_id,
                event.detection_count, event.peak_confidence,
            )
        else:
            event = self._create_event(session, detection)
            logger.debug(
                "Created event id=%d species_id=%d",
                event.id, detection.species_id,
            )

        detection.grouped_event_id = event.id
        session.flush()
        return event

    def close_stale_events(
        self,
        session,
        reference_time: Optional[datetime] = None,
    ) -> int:
        """
        Explicitly close DetectionEvents that have not received a new
        detection within gap_seconds.

        Sets ended_at to the started_at time for single-detection events,
        or leaves ended_at as-is (already set to last detection time)
        for multi-detection events that have gone quiet.

        Returns the number of events closed.
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        reference_naive = _as_naive_utc(reference_time)
        cutoff_naive = reference_naive - timedelta(seconds=self._gap_seconds)

        # All events that are not explicitly closed yet:
        # Either ended_at is None (single detection) or ended_at is old.
        # We identify stale ones by checking started_at and ended_at.
        all_events = session.query(DetectionEvent).all()

        closed = 0
        for event in all_events:
            # Determine last activity time
            if event.ended_at is not None:
                # Already has an ended_at — check if it's been explicitly
                # finalized (we treat events with ended_at as closed if
                # ended_at < cutoff and detection_count reflects reality)
                last_naive = _as_naive_utc(event.ended_at)
                if last_naive > cutoff_naive:
                    # Still active — skip
                    continue
                # Already closed naturally by _extend_event, skip re-closing
                continue

            # ended_at is None → single-detection event, use started_at
            started_naive = _as_naive_utc(event.started_at)
            if started_naive <= cutoff_naive:
                event.ended_at = event.started_at
                session.flush()
                closed += 1
                logger.debug("Closed stale event id=%d", event.id)

        if closed:
            logger.info("Closed %d stale detection event(s).", closed)
        return closed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_active_event(
        self,
        session,
        species_id: int,
        detection_timestamp: datetime,
    ) -> Optional[DetectionEvent]:
        """
        Return the most recent active DetectionEvent for *species_id*.

        An event is active if the time since its last detection is <=
        gap_seconds. Last detection time is:
          - ended_at  if it has multiple detections (ended_at is set by
                      _extend_event to the most recent detection time)
          - started_at if it has only one detection (ended_at is None)

        We query all events for this species and check in Python to
        avoid SQLite datetime comparison issues.
        """
        detection_naive = _as_naive_utc(detection_timestamp)
        cutoff_naive = detection_naive - timedelta(seconds=self._gap_seconds)

        # Get all events for this species, newest first
        events = (
            session.query(DetectionEvent)
            .filter(DetectionEvent.species_id == species_id)
            .order_by(DetectionEvent.started_at.desc())
            .all()
        )

        for event in events:
            # Determine last activity time for this event
            if event.ended_at is not None:
                last_naive = _as_naive_utc(event.ended_at)
            else:
                # Single-detection event: last activity = started_at
                last_naive = _as_naive_utc(event.started_at)

            # Skip events that are too old
            if last_naive < cutoff_naive:
                # Since events are ordered newest first, once we hit one
                # that's too old all subsequent ones are too
                break

            gap = (detection_naive - last_naive).total_seconds()
            if gap <= self._gap_seconds:
                return event

        return None

    def _extend_event(
        self,
        session,
        event: DetectionEvent,
        detection: Detection,
    ) -> DetectionEvent:
        """
        Update an existing event to include the new detection.

        ended_at is set to the detection timestamp to track last activity.
        This means ended_at is NOT the same as "the event is finished" —
        it is simply the timestamp of the most recent detection seen.
        """
        event.detection_count += 1
        event.peak_confidence = max(
            event.peak_confidence, detection.confidence
        )
        event.ended_at = detection.timestamp
        session.flush()
        return event

    def _create_event(
        self,
        session,
        detection: Detection,
    ) -> DetectionEvent:
        """Create and persist a new DetectionEvent."""
        return self._repo.add_detection_event(
            session,
            species_id=detection.species_id,
            started_at=detection.timestamp,
            peak_confidence=detection.confidence,
            detection_count=1,
            ended_at=None,  # signals single-detection event
        )
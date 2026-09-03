"""
BirdNetIdentifier — wraps birdnetlib behind the BirdIdentifier interface.

This is the real identification backend. It uses the BirdNET-Analyzer
model (via birdnetlib) to identify bird species in audio segments.

birdnetlib's RecordingBuffer class accepts an in-memory numpy array,
which plugs directly into our pipeline without writing files to disk.

RecordingBuffer signature (confirmed from installed version):
    RecordingBuffer(analyzer, buffer, rate, week_48=-1, date=None,
                    sensitivity=1.0, lat=None, lon=None, min_conf=0.1,
                    overlap=0.0, return_all_detections=False,
                    filter_threshold=0.03)

Key facts about BirdNET:
  - Expects audio at 48 000 Hz (librosa resamples internally if needed)
  - Designed for 3-second segments
  - Returns confidence scores in [0, 1]
  - Covers 6 000+ species including full European/Romanian coverage
  - Runs entirely offline after first model download
  - CPU-only inference is viable

Usage:

    from backend.identification.birdnet_identifier import BirdNetIdentifier

    identifier = BirdNetIdentifier()
    identifier.warmup()  # loads model weights once

    results = identifier.identify(segment, sample_rate=48000)
    for r in results:
        print(r.scientific_name, r.confidence)
"""

import sys
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.identification.base import (
    BirdIdentifier,
    BirdIdentifierError,
    IdentificationResult,
)

logger = logging.getLogger(__name__)

# Suppress the pydub ffmpeg warning — cosmetic only
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

# BirdNET expects audio at this sample rate
BIRDNET_SAMPLE_RATE = 48_000

# BirdNET model version we are using
BIRDNET_MODEL_VERSION = "2.4"

# Import at module level so tests can patch it
try:
    from birdnetlib import RecordingBuffer
    from birdnetlib.analyzer import Analyzer as _BirdnetAnalyzer
    _BIRDNETLIB_AVAILABLE = True
except ImportError:
    RecordingBuffer = None  # type: ignore
    _BirdnetAnalyzer = None  # type: ignore
    _BIRDNETLIB_AVAILABLE = False


class BirdNetIdentifier(BirdIdentifier):
    """
    BirdIdentifier implementation backed by BirdNET-Analyzer via birdnetlib.

    Parameters
    ----------
    latitude : float | None
        Observer latitude. When provided together with longitude,
        birdnetlib filters results to species predicted present at that
        location. Romania is approximately lat=45.9, lon=24.9.
    longitude : float | None
        Observer longitude. See latitude.
    week : int | None
        Week of year (1–48). Used for seasonal filtering when latitude
        and longitude are also provided. None disables week filtering.
    min_confidence : float
        Minimum confidence passed to birdnetlib's internal filter.
        Default 0.1 (permissive — let DetectionService decide).
    sensitivity : float
        Detection sensitivity passed to birdnetlib (0.5–1.5).
        Higher values detect more but may increase false positives.
        Default 1.0.
    """

    MODEL_NAME = "BirdNET"
    MODEL_VERSION = BIRDNET_MODEL_VERSION

    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        week: Optional[int] = None,
        min_confidence: float = 0.1,
        sensitivity: float = 1.0,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise BirdIdentifierError(
                f"min_confidence must be in [0, 1], got {min_confidence}."
            )

        self._latitude = latitude
        self._longitude = longitude
        self._week = week
        self._min_confidence = min_confidence
        self._sensitivity = sensitivity
        self._analyzer = None   # loaded lazily in warmup()
        self._available = False

        logger.debug(
            "BirdNetIdentifier created: lat=%s lon=%s week=%s min_conf=%.2f",
            latitude, longitude, week, min_confidence,
        )

    # ------------------------------------------------------------------
    # BirdIdentifier interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    @property
    def is_available(self) -> bool:
        return self._available

    def warmup(self) -> None:
        """
        Load the BirdNET model weights.

        Called once by DetectionService.start() before the pipeline
        begins. The first call downloads model weights if not cached
        (~100 MB, one-time only).

        Raises BirdIdentifierError if birdnetlib is not installed or
        the model cannot be loaded.
        """
        if self._analyzer is not None:
            return  # already warmed up

        if not _BIRDNETLIB_AVAILABLE:
            raise BirdIdentifierError(
                "birdnetlib is not installed. Run: pip install birdnetlib"
            )

        logger.info("Loading BirdNET model (first run may download weights)…")
        try:
            self._analyzer = _BirdnetAnalyzer()
            self._available = True
            logger.info("BirdNET model loaded successfully.")
        except Exception as exc:
            raise BirdIdentifierError(
                f"Failed to load BirdNET model: {exc}"
            ) from exc

    def identify(
        self,
        segment: np.ndarray,
        sample_rate: int,
        top_n: int = 5,
    ) -> list[IdentificationResult]:
        """
        Identify bird species in an audio segment using BirdNET.

        Parameters
        ----------
        segment : np.ndarray
            1-D float32 audio samples.
        sample_rate : int
            Sample rate of *segment* in Hz. Resampled to 48 000 Hz
            internally by birdnetlib/librosa if needed.
        top_n : int
            Maximum number of results to return, sorted by confidence
            descending.

        Returns
        -------
        list[IdentificationResult]
            Sorted by confidence descending. Empty if no species
            detected above min_confidence.

        Raises
        ------
        BirdIdentifierError
            If warmup() has not been called, or inference fails.
        """
        self._validate_input(segment, sample_rate)

        if self._analyzer is None:
            raise BirdIdentifierError(
                "BirdNetIdentifier.warmup() must be called before identify()."
            )

        timestamp = datetime.now(timezone.utc)
        duration = len(segment) / sample_rate

        try:
            results = self._run_inference(segment, sample_rate, duration, timestamp)
        except BirdIdentifierError:
            raise
        except Exception as exc:
            raise BirdIdentifierError(
                f"BirdNET inference failed: {exc}"
            ) from exc

        # Sort descending and apply top_n cap
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        segment: np.ndarray,
        sample_rate: int,
        duration: float,
        timestamp: datetime,
    ) -> list[IdentificationResult]:
        """
        Run birdnetlib inference on a numpy array segment.

        RecordingBuffer parameters (confirmed from installed version):
            buffer  — numpy array (float32, mono)
            rate    — sample rate in Hz
            week_48 — week of year 1-48, -1 to disable
            lat/lon — location filtering
            min_conf — minimum confidence
            sensitivity — detection sensitivity
        """
        kwargs: dict = {
            "min_conf": self._min_confidence,
            "sensitivity": self._sensitivity,
            "week_48": self._week if self._week is not None else -1,
        }
        if self._latitude is not None and self._longitude is not None:
            kwargs["lat"] = self._latitude
            kwargs["lon"] = self._longitude

        recording = RecordingBuffer(
            self._analyzer,
            buffer=segment,
            rate=sample_rate,
            **kwargs,
        )
        recording.analyze()

        results = []
        for detection in recording.detections:
            try:
                result = IdentificationResult(
                    scientific_name=detection["scientific_name"],
                    common_name=detection["common_name"],
                    confidence=float(detection["confidence"]),
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    timestamp=timestamp,
                    duration_seconds=duration,
                    raw_score=float(detection["confidence"]),
                )
                results.append(result)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed detection %r: %s", detection, exc
                )

        return results

    def _validate_input(self, segment: np.ndarray, sample_rate: int) -> None:
        if not isinstance(segment, np.ndarray):
            raise BirdIdentifierError(
                f"segment must be a numpy ndarray, got {type(segment).__name__}."
            )
        if segment.ndim != 1:
            raise BirdIdentifierError(
                f"segment must be 1-D, got shape {segment.shape}."
            )
        if len(segment) == 0:
            raise BirdIdentifierError("segment must not be empty.")
        if sample_rate <= 0:
            raise BirdIdentifierError(
                f"sample_rate must be positive, got {sample_rate}."
            )
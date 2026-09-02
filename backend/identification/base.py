"""
BirdIdentifier — abstract base class for all bird identification backends.

Every identifier implementation (mock, BirdNET, future alternatives)
must subclass BirdIdentifier and implement identify().

The rest of the application only imports from this module, never from
a concrete backend directly. Switching backends is a one-line config
change.

Data flow:

    AudioProcessor segment
        └─► BirdIdentifier.identify(segment, sample_rate)
                └─► list[IdentificationResult]
                        └─► DetectionService  (filters by confidence,
                                               persists to database)

Design notes:
  - identify() returns a LIST because some models (BirdNET included)
    return ranked top-N candidates for a single audio segment.
  - Results are sorted by confidence descending.
  - An empty list means "no species detected above the model's internal
    threshold" — it is NOT an error.
  - Errors in the underlying model raise BirdIdentifierError.
"""

import sys
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class IdentificationResult:
    """
    A single species candidate returned by a BirdIdentifier.

    Attributes
    ----------
    scientific_name : str
        Latin binomial, e.g. "Erithacus rubecula".
        Used as the stable unique key throughout the system.
    common_name : str
        Display name, e.g. "European Robin".
    confidence : float
        Model confidence in [0.0, 1.0].  Higher is more certain.
    model_name : str
        Identifier string for the model, e.g. "BirdNET" or "Mock".
    model_version : str
        Version string, e.g. "2.4" or "0.1".
    timestamp : datetime
        UTC datetime when the segment was analysed (not when the bird
        called — that is the segment start time, tracked separately).
    duration_seconds : float | None
        Length of the audio segment that was classified, in seconds.
    raw_score : float | None
        The model's raw output score before any normalisation, if
        available. Useful for debugging and model comparison.
    """
    scientific_name: str
    common_name: str
    confidence: float
    model_name: str
    model_version: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_seconds: Optional[float] = None
    raw_score: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}."
            )
        if not self.scientific_name.strip():
            raise ValueError("scientific_name must not be empty.")
        if not self.common_name.strip():
            raise ValueError("common_name must not be empty.")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError(
                f"duration_seconds must be positive, got {self.duration_seconds}."
            )

    @property
    def is_confident(self, threshold: float = 0.5) -> bool:
        """True if confidence meets the default threshold."""
        return self.confidence >= threshold

    def above_threshold(self, threshold: float) -> bool:
        """True if confidence >= threshold."""
        return self.confidence >= threshold

    def __repr__(self) -> str:
        return (
            f"<IdentificationResult {self.scientific_name!r} "
            f"confidence={self.confidence:.2f} model={self.model_name}>"
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BirdIdentifier(ABC):
    """
    Abstract base class for bird audio identification backends.

    Subclasses must implement:
        identify(segment, sample_rate) -> list[IdentificationResult]

    Subclasses may optionally override:
        model_name    -> str   (property)
        model_version -> str   (property)
        is_available  -> bool  (property)
        warmup()               (called once before first use)
    """

    # ------------------------------------------------------------------
    # Properties subclasses should override
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Human-readable model name, e.g. 'BirdNET'."""
        return self.__class__.__name__

    @property
    def model_version(self) -> str:
        """Version string for the model, e.g. '2.4'."""
        return "unknown"

    @property
    def is_available(self) -> bool:
        """
        True if the backend is ready to accept identify() calls.

        Subclasses may override this to check, for example, that model
        weights have been downloaded or that a required library is
        installed.
        """
        return True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def identify(
        self,
        segment: "np.ndarray",  # noqa: F821  (imported in subclasses)
        sample_rate: int,
        top_n: int = 5,
    ) -> list[IdentificationResult]:
        """
        Identify bird species in an audio segment.

        Parameters
        ----------
        segment : np.ndarray
            1-D float32 audio samples.
        sample_rate : int
            Sample rate of *segment* in Hz.
        top_n : int
            Maximum number of candidate results to return.
            Results are sorted by confidence descending.
            Fewer than top_n results may be returned if the model has
            fewer confident candidates.

        Returns
        -------
        list[IdentificationResult]
            Sorted by confidence descending. Empty list if no species
            detected above the model's internal threshold.

        Raises
        ------
        BirdIdentifierError
            If the model cannot process the segment (e.g. wrong sample
            rate, model not loaded, internal inference error).
        """
        ...

    def identify_and_filter(
        self,
        segment: "np.ndarray",  # noqa: F821
        sample_rate: int,
        min_confidence: float = 0.5,
        top_n: int = 5,
    ) -> list[IdentificationResult]:
        """
        Convenience wrapper: identify() then filter by min_confidence.

        Returns only results with confidence >= min_confidence,
        still sorted descending.
        """
        results = self.identify(segment, sample_rate, top_n=top_n)
        return [r for r in results if r.confidence >= min_confidence]

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """
        Optional: pre-load model weights or run a dummy inference.

        Called once by DetectionService before the pipeline starts.
        Default implementation does nothing.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"model={self.model_name!r} version={self.model_version!r} "
            f"available={self.is_available}>"
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BirdIdentifierError(Exception):
    """
    Raised when a BirdIdentifier cannot process a segment.

    Examples:
      - Model weights not found.
      - Segment has wrong sample rate for this model.
      - Internal inference failure.
    """
    pass
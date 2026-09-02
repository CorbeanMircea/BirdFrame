"""
MockBirdIdentifier — a configurable fake bird identifier for development
and testing.

Returns realistic-looking IdentificationResult objects without loading
any model or processing any audio. The audio segment is accepted but
ignored.

Designed for:
  - Running the full pipeline without a real model installed.
  - Deterministic tests that need predictable identification output.
  - Demonstrating the UI / database / collage with realistic species data.

Behaviour is controlled by the `mode` parameter:

  "random"      — Pick species randomly from the built-in list on every
                  call. Confidences are also randomised within a
                  configurable range. Good for demos.

  "fixed"       — Always return the same species list in the same order.
                  Confidence values are fixed. Good for unit tests.

  "sequential"  — Cycle through the species list in order, returning one
                  species per call. Good for testing detection grouping
                  and history behaviour.

  "empty"       — Always return an empty list (no detections). Good for
                  testing the "nothing heard" code path.

Usage:

    from backend.identification.mock_identifier import MockBirdIdentifier

    # Random mode (default) — good for demos
    identifier = MockBirdIdentifier()
    results = identifier.identify(segment, sample_rate=48000)

    # Fixed mode — good for tests
    identifier = MockBirdIdentifier(mode="fixed")
    results = identifier.identify(segment, sample_rate=48000)
    assert results[0].scientific_name == "Erithacus rubecula"
"""

import sys
import random
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.identification.base import (
    BirdIdentifier,
    BirdIdentifierError,
    IdentificationResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in species list — common European / Romanian birds
# ---------------------------------------------------------------------------

_DEFAULT_SPECIES: list[tuple[str, str]] = [
    ("Erithacus rubecula",      "European Robin"),
    ("Parus major",             "Great Tit"),
    ("Turdus merula",           "Eurasian Blackbird"),
    ("Passer domesticus",       "House Sparrow"),
    ("Fringilla coelebs",       "Common Chaffinch"),
    ("Cyanistes caeruleus",     "Blue Tit"),
    ("Sylvia atricapilla",      "Eurasian Blackcap"),
    ("Carduelis carduelis",     "European Goldfinch"),
    ("Columba palumbus",        "Common Wood Pigeon"),
    ("Hirundo rustica",         "Barn Swallow"),
    ("Apus apus",               "Common Swift"),
    ("Cuculus canorus",         "Common Cuckoo"),
    ("Dendrocopos major",       "Great Spotted Woodpecker"),
    ("Sitta europaea",          "Eurasian Nuthatch"),
    ("Troglodytes troglodytes", "Eurasian Wren"),
    ("Motacilla alba",          "White Wagtail"),
    ("Garrulus glandarius",     "Eurasian Jay"),
    ("Pica pica",               "Eurasian Magpie"),
    ("Corvus cornix",           "Hooded Crow"),
    ("Oriolus oriolus",         "Eurasian Golden Oriole"),
]

MockMode = Literal["random", "fixed", "sequential", "empty"]


class MockBirdIdentifier(BirdIdentifier):
    """
    Configurable fake bird identifier for development and testing.

    Parameters
    ----------
    mode : MockMode
        Controls what results are returned. See module docstring.
    species_list : list[tuple[str, str]] | None
        List of (scientific_name, common_name) pairs to draw from.
        Defaults to _DEFAULT_SPECIES (20 common European birds).
    fixed_confidence : float
        Confidence returned in "fixed" and "sequential" modes.
        Default 0.85.
    confidence_range : tuple[float, float]
        (min, max) confidence range used in "random" mode.
        Default (0.50, 0.99).
    seed : int | None
        Random seed for reproducible "random" mode output.
        None means non-deterministic (true random).
    call_delay : float
        Simulated inference delay in seconds. 0 by default (instant).
        Set to a small value to simulate model latency in demos.
    """

    MODEL_NAME = "MockBirdIdentifier"
    MODEL_VERSION = "0.1"

    def __init__(
        self,
        mode: MockMode = "random",
        species_list: list[tuple[str, str]] | None = None,
        fixed_confidence: float = 0.85,
        confidence_range: tuple[float, float] = (0.50, 0.99),
        seed: int | None = None,
        call_delay: float = 0.0,
    ) -> None:
        if mode not in ("random", "fixed", "sequential", "empty"):
            raise BirdIdentifierError(
                f"Unknown mode {mode!r}. "
                "Choose from: 'random', 'fixed', 'sequential', 'empty'."
            )
        if not 0.0 <= fixed_confidence <= 1.0:
            raise BirdIdentifierError(
                f"fixed_confidence must be in [0, 1], got {fixed_confidence}."
            )
        lo, hi = confidence_range
        if not (0.0 <= lo <= hi <= 1.0):
            raise BirdIdentifierError(
                f"confidence_range must be [lo, hi] with 0 ≤ lo ≤ hi ≤ 1, "
                f"got {confidence_range}."
            )
        if call_delay < 0:
            raise BirdIdentifierError(
                f"call_delay must be >= 0, got {call_delay}."
            )

        self._mode = mode
        self._species = species_list if species_list is not None else _DEFAULT_SPECIES
        self._fixed_confidence = fixed_confidence
        self._confidence_range = confidence_range
        self._call_delay = call_delay
        self._rng = random.Random(seed)
        self._call_count = 0          # used by "sequential" mode
        self._sequential_index = 0

        if not self._species:
            raise BirdIdentifierError("species_list must not be empty.")

        logger.debug(
            "MockBirdIdentifier initialised: mode=%r species_count=%d",
            mode, len(self._species),
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
        return True

    def warmup(self) -> None:
        logger.debug("MockBirdIdentifier.warmup() called — nothing to do.")

    def identify(
        self,
        segment: np.ndarray,
        sample_rate: int,
        top_n: int = 5,
    ) -> list[IdentificationResult]:
        """
        Return fake identification results according to the current mode.

        The segment is validated for shape but its values are ignored.
        """
        self._validate_input(segment, sample_rate)

        if self._call_delay > 0:
            import time
            time.sleep(self._call_delay)

        self._call_count += 1
        duration = len(segment) / sample_rate

        if self._mode == "empty":
            return []

        if self._mode == "fixed":
            return self._fixed_results(top_n, duration)

        if self._mode == "sequential":
            return self._sequential_result(duration)

        # default: "random"
        return self._random_results(top_n, duration)

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _fixed_results(
        self, top_n: int, duration: float
    ) -> list[IdentificationResult]:
        """
        Always return the first top_n species with fixed_confidence,
        in species_list order (descending confidence with tiny offsets
        so ordering is stable and testable).
        """
        results = []
        count = min(top_n, len(self._species))
        for i, (sci, common) in enumerate(self._species[:count]):
            # Tiny decrement so results are strictly ordered
            confidence = max(0.0, self._fixed_confidence - i * 0.01)
            results.append(
                IdentificationResult(
                    scientific_name=sci,
                    common_name=common,
                    confidence=round(confidence, 4),
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    timestamp=datetime.now(timezone.utc),
                    duration_seconds=duration,
                )
            )
        return results  # already descending

    def _sequential_result(self, duration: float) -> list[IdentificationResult]:
        """
        Return one species per call, cycling through the species list.
        Each call advances to the next species.
        """
        sci, common = self._species[self._sequential_index]
        self._sequential_index = (self._sequential_index + 1) % len(self._species)
        return [
            IdentificationResult(
                scientific_name=sci,
                common_name=common,
                confidence=self._fixed_confidence,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                timestamp=datetime.now(timezone.utc),
                duration_seconds=duration,
            )
        ]

    def _random_results(
        self, top_n: int, duration: float
    ) -> list[IdentificationResult]:
        """
        Pick top_n species at random (without replacement if possible),
        assign random confidences, and sort descending.
        """
        count = min(top_n, len(self._species))
        chosen = self._rng.sample(self._species, count)
        lo, hi = self._confidence_range

        results = []
        for sci, common in chosen:
            confidence = round(self._rng.uniform(lo, hi), 4)
            results.append(
                IdentificationResult(
                    scientific_name=sci,
                    common_name=common,
                    confidence=confidence,
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    timestamp=datetime.now(timezone.utc),
                    duration_seconds=duration,
                )
            )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    @property
    def call_count(self) -> int:
        """Number of times identify() has been called."""
        return self._call_count

    def reset(self) -> None:
        """Reset call counter and sequential index (useful between tests)."""
        self._call_count = 0
        self._sequential_index = 0
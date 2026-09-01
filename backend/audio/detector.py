"""
BirdDetector — lightweight pre-filter that decides whether an audio
segment is worth sending to the bird identifier.

The detector answers a single binary question:
    "Does this segment contain sounds that could be bird vocalisations?"

It uses two fast, cheap signal-processing heuristics:

1. RMS energy threshold  — reject silence and very quiet segments.
2. Spectral band energy  — check whether energy exists in the frequency
   range where most bird vocalisations occur (default 1 000–10 000 Hz).
   Segments dominated by low-frequency noise (traffic, wind rumble) are
   rejected even if they are loud.

Both checks must pass for a segment to be accepted.

This is intentionally conservative (low false-negative rate): it is
better to send a few non-bird segments to the identifier than to miss
real bird calls. The identifier itself applies a confidence threshold
to filter weak results.

The detector is entirely stateless — every call to `is_bird_audio()`
is independent.

Usage:

    detector = BirdDetector()
    if detector.is_bird_audio(segment, sample_rate=48000):
        result = identifier.identify(segment, sample_rate)
"""

import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


@dataclass
class DetectorResult:
    """
    Full diagnostic result from a single detector pass.

    Attributes
    ----------
    accepted : bool
        True if the segment passed all checks and should be identified.
    rms_energy : float
        Root-mean-square energy of the full segment (0.0–1.0 range for
        normalised float32 audio).
    band_energy_ratio : float
        Fraction of total spectral power that falls within the bird
        frequency band [freq_min, freq_max].
    rms_threshold : float
        The threshold that rms_energy was compared against.
    band_ratio_threshold : float
        The threshold that band_energy_ratio was compared against.
    rejection_reason : str | None
        Human-readable reason if accepted is False, else None.
    """
    accepted: bool
    rms_energy: float
    band_energy_ratio: float
    rms_threshold: float
    band_ratio_threshold: float
    rejection_reason: Optional[str] = None


class BirdDetectorError(Exception):
    """Raised when the detector receives invalid input."""
    pass


class BirdDetector:
    """
    Energy- and frequency-based pre-filter for bird audio detection.

    Parameters
    ----------
    energy_threshold : float
        Minimum RMS energy for a segment to be considered non-silent.
        Defaults to config.DETECTOR_ENERGY_THRESHOLD.
    freq_min : int
        Lower bound of the bird-frequency band in Hz.
        Defaults to config.DETECTOR_FREQ_MIN.
    freq_max : int
        Upper bound of the bird-frequency band in Hz.
        Defaults to config.DETECTOR_FREQ_MAX.
    band_ratio_threshold : float
        Minimum fraction of spectral energy that must fall within
        [freq_min, freq_max] for the segment to be accepted.
        Range 0.0–1.0. Default 0.05 (5 %) — deliberately permissive.
    """

    def __init__(
        self,
        energy_threshold: float = config.DETECTOR_ENERGY_THRESHOLD,
        freq_min: int = config.DETECTOR_FREQ_MIN,
        freq_max: int = config.DETECTOR_FREQ_MAX,
        band_ratio_threshold: float = 0.05,
    ) -> None:
        if freq_min >= freq_max:
            raise BirdDetectorError(
                f"freq_min ({freq_min}) must be less than freq_max ({freq_max})."
            )
        if not 0.0 <= band_ratio_threshold <= 1.0:
            raise BirdDetectorError(
                f"band_ratio_threshold must be in [0, 1], got {band_ratio_threshold}."
            )
        if energy_threshold < 0:
            raise BirdDetectorError(
                f"energy_threshold must be >= 0, got {energy_threshold}."
            )

        self._energy_threshold = energy_threshold
        self._freq_min = freq_min
        self._freq_max = freq_max
        self._band_ratio_threshold = band_ratio_threshold

        logger.debug(
            "BirdDetector initialised: energy_threshold=%.4f "
            "freq_band=%d–%d Hz band_ratio_threshold=%.2f",
            energy_threshold, freq_min, freq_max, band_ratio_threshold,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def energy_threshold(self) -> float:
        return self._energy_threshold

    @property
    def freq_min(self) -> int:
        return self._freq_min

    @property
    def freq_max(self) -> int:
        return self._freq_max

    @property
    def band_ratio_threshold(self) -> float:
        return self._band_ratio_threshold

    def is_bird_audio(self, segment: np.ndarray, sample_rate: int) -> bool:
        """
        Return True if *segment* passes both the energy and band checks.

        This is the fast path used in the live pipeline.
        For debugging, use analyse() to get the full DetectorResult.

        Parameters
        ----------
        segment : np.ndarray
            1-D float32 audio segment.
        sample_rate : int
            Sample rate of *segment* in Hz.
        """
        return self.analyse(segment, sample_rate).accepted

    def analyse(self, segment: np.ndarray, sample_rate: int) -> DetectorResult:
        """
        Run all checks and return a full DetectorResult with diagnostics.

        Parameters
        ----------
        segment : np.ndarray
            1-D float32 audio segment.
        sample_rate : int
            Sample rate of *segment* in Hz.
        """
        self._validate(segment, sample_rate)

        rms = _compute_rms(segment)

        # --- Check 1: RMS energy ---
        if rms < self._energy_threshold:
            logger.debug(
                "Segment rejected: rms=%.5f < threshold=%.5f", rms, self._energy_threshold
            )
            return DetectorResult(
                accepted=False,
                rms_energy=rms,
                band_energy_ratio=0.0,
                rms_threshold=self._energy_threshold,
                band_ratio_threshold=self._band_ratio_threshold,
                rejection_reason=(
                    f"RMS energy {rms:.5f} below threshold {self._energy_threshold:.5f}"
                ),
            )

        # --- Check 2: Spectral band energy ---
        band_ratio = _compute_band_energy_ratio(
            segment, sample_rate, self._freq_min, self._freq_max
        )

        if band_ratio < self._band_ratio_threshold:
            logger.debug(
                "Segment rejected: band_ratio=%.3f < threshold=%.3f",
                band_ratio, self._band_ratio_threshold,
            )
            return DetectorResult(
                accepted=False,
                rms_energy=rms,
                band_energy_ratio=band_ratio,
                rms_threshold=self._energy_threshold,
                band_ratio_threshold=self._band_ratio_threshold,
                rejection_reason=(
                    f"Band energy ratio {band_ratio:.3f} below "
                    f"threshold {self._band_ratio_threshold:.3f}"
                ),
            )

        logger.debug(
            "Segment accepted: rms=%.5f band_ratio=%.3f", rms, band_ratio
        )
        return DetectorResult(
            accepted=True,
            rms_energy=rms,
            band_energy_ratio=band_ratio,
            rms_threshold=self._energy_threshold,
            band_ratio_threshold=self._band_ratio_threshold,
            rejection_reason=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, segment: np.ndarray, sample_rate: int) -> None:
        if not isinstance(segment, np.ndarray):
            raise BirdDetectorError(
                f"segment must be a numpy ndarray, got {type(segment).__name__}."
            )
        if segment.ndim != 1:
            raise BirdDetectorError(
                f"segment must be 1-D, got shape {segment.shape}."
            )
        if len(segment) == 0:
            raise BirdDetectorError("segment must not be empty.")
        if sample_rate <= 0:
            raise BirdDetectorError(
                f"sample_rate must be positive, got {sample_rate}."
            )
        if self._freq_max > sample_rate / 2:
            logger.warning(
                "freq_max (%d Hz) exceeds Nyquist frequency (%d Hz) for "
                "sample_rate=%d. Band check may be inaccurate.",
                self._freq_max, sample_rate // 2, sample_rate,
            )


# ---------------------------------------------------------------------------
# Signal processing utilities
# ---------------------------------------------------------------------------

def _compute_rms(segment: np.ndarray) -> float:
    """
    Compute the root-mean-square energy of a 1-D audio segment.

    Returns a float in [0, 1] for normalised float32 input.
    """
    return float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))


def _compute_band_energy_ratio(
    segment: np.ndarray,
    sample_rate: int,
    freq_min: int,
    freq_max: int,
) -> float:
    """
    Return the fraction of total spectral power within [freq_min, freq_max].

    Uses a real FFT. Returns 0.0 if total power is zero (silent segment).

    Parameters
    ----------
    segment : np.ndarray    1-D float32
    sample_rate : int
    freq_min : int          lower band edge in Hz
    freq_max : int          upper band edge in Hz
    """
    # Real FFT → only positive frequencies
    spectrum = np.fft.rfft(segment.astype(np.float64))
    power = np.abs(spectrum) ** 2

    total_power = power.sum()
    if total_power == 0.0:
        return 0.0

    # Frequency axis for the rfft output
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)

    band_mask = (freqs >= freq_min) & (freqs <= freq_max)
    band_power = power[band_mask].sum()

    return float(band_power / total_power)
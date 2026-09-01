"""
AudioProcessor — prepares raw audio chunks for bird detection.

Responsibilities:
  - Accept raw numpy chunks from AudioRecorder via process().
  - Maintain a sliding overlap buffer so sounds at chunk boundaries
    are never silently dropped.
  - Emit fixed-length, overlap-aware segments to a downstream callback.
  - Optionally resample to a target sample rate (needed when recorder
    sample rate differs from what BirdNET expects).

This class does NOT perform any bird detection or identification.
It only shapes the audio data for the next stage.

Data flow:

    AudioRecorder
        └─► AudioProcessor.process(chunk, sample_rate)
                └─► [overlap + resample if needed]
                        └─► segment_callback(segment, sample_rate)
                                └─► BirdDetector  (next stage)

Usage example:

    def on_segment(segment: np.ndarray, sample_rate: int) -> None:
        print(f"segment ready: {len(segment)} samples @ {sample_rate} Hz")

    processor = AudioProcessor(segment_callback=on_segment)

    # Feed chunks as they arrive from AudioRecorder:
    processor.process(chunk, sample_rate=44100)
"""

import sys
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)

# Type alias
SegmentCallback = Callable[[np.ndarray, int], None]


class AudioProcessorError(Exception):
    """Raised when the AudioProcessor encounters an unrecoverable error."""
    pass


class AudioProcessor:
    """
    Converts a stream of raw audio chunks into overlap-aware segments.

    Parameters
    ----------
    segment_callback : SegmentCallback
        Called with (segment: np.ndarray, sample_rate: int) for every
        completed segment. The segment is always 1-D float32.
    segment_duration : float
        Desired segment length in seconds.
        Defaults to config.AUDIO_CHUNK_DURATION.
    overlap_duration : float
        How many seconds of the previous segment to prepend to the next.
        Must be strictly less than segment_duration.
        Defaults to config.AUDIO_CHUNK_OVERLAP.
    target_sample_rate : int | None
        If set, segments are resampled to this rate before the callback
        is invoked. None means no resampling (pass-through).
        Useful when recorder runs at 44100 but BirdNET expects 48000.
    """

    def __init__(
        self,
        segment_callback: SegmentCallback,
        segment_duration: float = config.AUDIO_CHUNK_DURATION,
        overlap_duration: float = config.AUDIO_CHUNK_OVERLAP,
        target_sample_rate: Optional[int] = None,
    ) -> None:
        if overlap_duration >= segment_duration:
            raise AudioProcessorError(
                f"overlap_duration ({overlap_duration}s) must be less than "
                f"segment_duration ({segment_duration}s)."
            )

        self._callback = segment_callback
        self._segment_duration = segment_duration
        self._overlap_duration = overlap_duration
        self._target_sample_rate = target_sample_rate

        # These are computed once we receive the first chunk and know
        # the incoming sample rate.
        self._sample_rate: Optional[int] = None
        self._segment_samples: Optional[int] = None
        self._overlap_samples: Optional[int] = None
        self._step_samples: Optional[int] = None

        # Ring buffer: accumulates samples between process() calls.
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)

        logger.debug(
            "AudioProcessor initialised: segment=%.1fs overlap=%.1fs target_sr=%s",
            segment_duration,
            overlap_duration,
            target_sample_rate,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> Optional[int]:
        """The input sample rate, set on first call to process()."""
        return self._sample_rate

    @property
    def segment_samples(self) -> Optional[int]:
        """Number of samples per emitted segment (at input sample rate)."""
        return self._segment_samples

    def process(self, chunk: np.ndarray, sample_rate: int) -> None:
        """
        Accept a new chunk of audio and emit any complete segments.

        Parameters
        ----------
        chunk : np.ndarray
            1-D float32 array of audio samples.
        sample_rate : int
            Sample rate of *chunk* in Hz.
        """
        chunk = self._validate_chunk(chunk)

        # Initialise sample-count parameters on first call.
        if self._sample_rate is None:
            self._initialise(sample_rate)
        elif sample_rate != self._sample_rate:
            raise AudioProcessorError(
                f"Sample rate changed mid-stream: expected {self._sample_rate} "
                f"but received {sample_rate}."
            )

        self._buffer = np.concatenate([self._buffer, chunk])

        # Emit segments while buffer holds at least one full segment.
        while len(self._buffer) >= self._segment_samples:
            segment = self._buffer[: self._segment_samples].copy()
            # Advance by step (segment - overlap), keeping overlap tail.
            self._buffer = self._buffer[self._step_samples :]
            self._emit(segment, self._sample_rate)

    def reset(self) -> None:
        """
        Clear the internal buffer and reset sample-rate state.

        Call this if the audio stream is restarted with potentially
        different parameters.
        """
        self._buffer = np.empty(0, dtype=np.float32)
        self._sample_rate = None
        self._segment_samples = None
        self._overlap_samples = None
        self._step_samples = None
        logger.debug("AudioProcessor reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialise(self, sample_rate: int) -> None:
        """Set all sample-count fields from the known sample rate."""
        self._sample_rate = sample_rate
        self._segment_samples = int(sample_rate * self._segment_duration)
        self._overlap_samples = int(sample_rate * self._overlap_duration)
        self._step_samples = self._segment_samples - self._overlap_samples
        logger.debug(
            "AudioProcessor configured: sr=%d segment=%d overlap=%d step=%d",
            sample_rate,
            self._segment_samples,
            self._overlap_samples,
            self._step_samples,
        )

    def _validate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Ensure chunk is a 1-D float32 array."""
        if not isinstance(chunk, np.ndarray):
            raise AudioProcessorError(
                f"chunk must be a numpy ndarray, got {type(chunk).__name__}."
            )
        if chunk.ndim != 1:
            raise AudioProcessorError(
                f"chunk must be 1-D, got shape {chunk.shape}. "
                "Use mono audio or squeeze before passing."
            )
        return chunk.astype(np.float32, copy=False)

    def _emit(self, segment: np.ndarray, input_sample_rate: int) -> None:
        """Resample if needed, then invoke the downstream callback."""
        if self._target_sample_rate is not None and \
                self._target_sample_rate != input_sample_rate:
            segment = _resample(segment, input_sample_rate, self._target_sample_rate)
            out_rate = self._target_sample_rate
        else:
            out_rate = input_sample_rate

        try:
            self._callback(segment, out_rate)
        except Exception as exc:  # noqa: BLE001
            logger.error("Exception in AudioProcessor segment callback: %s", exc)


# ---------------------------------------------------------------------------
# Resampling utility (pure numpy, no extra dependencies)
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """
    Resample a 1-D float32 audio array from from_rate to to_rate.

    Uses linear interpolation — sufficient quality for the analysis
    rates we work with (44100 → 48000 or vice-versa).
    For higher quality, swap this for scipy.signal.resample_poly later
    without changing the rest of the codebase.

    Parameters
    ----------
    audio : np.ndarray  1-D float32
    from_rate : int     source sample rate
    to_rate   : int     target sample rate

    Returns
    -------
    np.ndarray  1-D float32 at to_rate
    """
    if from_rate == to_rate:
        return audio

    target_length = int(len(audio) * to_rate / from_rate)
    if target_length == 0:
        return np.empty(0, dtype=np.float32)

    # Build the output sample positions in terms of input indices
    x_old = np.linspace(0, len(audio) - 1, len(audio))
    x_new = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(x_new, x_old, audio).astype(np.float32)
    return resampled
"""
MockAudioRecorder — feeds a WAV file (or synthetic audio) through the
same interface as AudioRecorder.

Used in tests and during development when no physical microphone is
available, or when reproducible audio input is needed.

The mock reads audio data and delivers it to the callback in the same
fixed-size chunk pattern that AudioRecorder uses, so downstream
components (AudioProcessor, BirdDetector) cannot tell the difference.

Usage example:

    from backend.audio.mock_recorder import MockAudioRecorder
    import numpy as np

    # From a WAV file:
    recorder = MockAudioRecorder.from_wav("path/to/bird.wav", callback=my_cb)
    recorder.start()   # non-blocking; runs in a background thread
    recorder.join()    # wait until all audio has been delivered

    # From a synthetic numpy array:
    audio = np.zeros(48000, dtype=np.float32)
    recorder = MockAudioRecorder.from_array(audio, sample_rate=48000, callback=my_cb)
    recorder.start()
    recorder.join()
"""

import sys
import wave
import struct
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)

ChunkCallback = Callable[[np.ndarray, int], None]


class MockAudioRecorderError(Exception):
    """Raised when MockAudioRecorder cannot load or play back audio."""
    pass


class MockAudioRecorder:
    """
    Replays audio data through the AudioRecorder callback interface.

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 audio samples to replay.
    sample_rate : int
        Sample rate of *audio*.
    callback : ChunkCallback
        Called with (chunk, sample_rate) for each chunk, identical to
        the real AudioRecorder contract.
    chunk_duration : float
        Size of each delivered chunk in seconds.
        Defaults to config.AUDIO_CHUNK_DURATION.
    realtime : bool
        If True, sleep between chunks to simulate real-time delivery.
        If False (default), deliver all chunks as fast as possible —
        useful in tests where speed matters.
    """

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        callback: ChunkCallback,
        chunk_duration: float = config.AUDIO_CHUNK_DURATION,
        realtime: bool = False,
    ) -> None:
        if audio.ndim != 1:
            raise MockAudioRecorderError(
                f"audio must be 1-D, got shape {audio.shape}."
            )
        self._audio = audio.astype(np.float32)
        self._sample_rate = sample_rate
        self._callback = callback
        self._chunk_samples = int(sample_rate * chunk_duration)
        self._realtime = realtime

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_wav(
        cls,
        path: str | Path,
        callback: ChunkCallback,
        chunk_duration: float = config.AUDIO_CHUNK_DURATION,
        realtime: bool = False,
    ) -> "MockAudioRecorder":
        """
        Create a MockAudioRecorder from a WAV file.

        Supports 8-bit, 16-bit, and 32-bit PCM WAV files.
        Multi-channel files are mixed down to mono.
        """
        audio, sample_rate = _load_wav(Path(path))
        return cls(
            audio=audio,
            sample_rate=sample_rate,
            callback=callback,
            chunk_duration=chunk_duration,
            realtime=realtime,
        )

    @classmethod
    def from_array(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        callback: ChunkCallback,
        chunk_duration: float = config.AUDIO_CHUNK_DURATION,
        realtime: bool = False,
    ) -> "MockAudioRecorder":
        """Create a MockAudioRecorder from a numpy array."""
        return cls(
            audio=audio,
            sample_rate=sample_rate,
            callback=callback,
            chunk_duration=chunk_duration,
            realtime=realtime,
        )

    # ------------------------------------------------------------------
    # Public interface (mirrors AudioRecorder)
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start playback in a background thread (non-blocking)."""
        if self._running:
            logger.warning("MockAudioRecorder.start() called while already running.")
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.debug("MockAudioRecorder started (%d samples).", len(self._audio))

    def stop(self) -> None:
        """Signal the playback thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False
        logger.debug("MockAudioRecorder stopped.")

    def join(self, timeout: Optional[float] = None) -> None:
        """Block until all audio has been delivered (or timeout expires)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def __enter__(self) -> "MockAudioRecorder":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Playback thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Deliver audio chunks to the callback, then set _running=False."""
        try:
            offset = 0
            total = len(self._audio)

            while offset < total and not self._stop_event.is_set():
                end = min(offset + self._chunk_samples, total)
                chunk = self._audio[offset:end]

                # Pad the final chunk with silence if shorter than chunk_samples
                if len(chunk) < self._chunk_samples:
                    chunk = np.pad(chunk, (0, self._chunk_samples - len(chunk)))

                try:
                    self._callback(chunk, self._sample_rate)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Exception in MockAudioRecorder callback: %s", exc)

                offset += self._chunk_samples

                if self._realtime:
                    chunk_secs = self._chunk_samples / self._sample_rate
                    time.sleep(chunk_secs)

        finally:
            self._running = False
            logger.debug("MockAudioRecorder playback complete.")


# ---------------------------------------------------------------------------
# WAV loading utility
# ---------------------------------------------------------------------------

def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and return (audio_float32_mono, sample_rate).

    Raises MockAudioRecorderError if the file cannot be read.
    """
    if not path.exists():
        raise MockAudioRecorderError(f"WAV file not found: {path}")

    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()  # bytes per sample
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except wave.Error as exc:
        raise MockAudioRecorderError(f"Cannot read WAV file {path}: {exc}") from exc

    # Decode raw bytes to numpy
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise MockAudioRecorderError(
            f"Unsupported sample width: {sample_width} bytes."
        )

    samples = np.frombuffer(raw, dtype=dtype)

    # Reshape to (n_frames, n_channels) and mix down to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
        samples = samples.mean(axis=1)

    # Normalise to float32 in [-1, 1]
    if sample_width == 1:
        # uint8: 0–255, centre at 128
        audio = (samples.astype(np.float32) - 128.0) / 128.0
    else:
        max_val = float(2 ** (8 * sample_width - 1))
        audio = samples.astype(np.float32) / max_val

    return audio, sample_rate


# ---------------------------------------------------------------------------
# WAV writing utility (used by tests to create synthetic WAV files)
# ---------------------------------------------------------------------------

def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """
    Write a 1-D float32 numpy array to a 16-bit mono WAV file.

    Useful for creating small test fixtures without external tools.
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
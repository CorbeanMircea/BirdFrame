"""
AudioRecorder — captures audio from a microphone using sounddevice.

Responsibilities:
  - Open an input stream from the configured (or default) device.
  - Deliver a continuous stream of fixed-size numpy chunks to a callback.
  - Provide a helper to list available input devices.
  - Be startable and stoppable cleanly.

This class does NOT process, analyse, or store audio.
That is the job of AudioProcessor and BirdDetector downstream.

Usage example:

    def handle_chunk(chunk: np.ndarray, sample_rate: int) -> None:
        print(f"Got chunk shape={chunk.shape}")

    recorder = AudioRecorder(callback=handle_chunk)
    recorder.start()
    time.sleep(10)
    recorder.stop()
"""

import sys
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)

# Type alias for the chunk callback
ChunkCallback = Callable[[np.ndarray, int], None]


class AudioRecorderError(Exception):
    """Raised when the AudioRecorder cannot open or maintain the stream."""
    pass


class AudioRecorder:
    """
    Captures audio from a microphone and delivers chunks to a callback.

    Parameters
    ----------
    callback : ChunkCallback
        Called with (chunk: np.ndarray, sample_rate: int) for every
        completed chunk. Runs on the sounddevice callback thread —
        keep it fast or hand off to a queue.
    sample_rate : int
        Samples per second. Defaults to config.AUDIO_SAMPLE_RATE.
    chunk_duration : float
        Length of each delivered chunk in seconds.
        Defaults to config.AUDIO_CHUNK_DURATION.
    channels : int
        Number of input channels. Defaults to config.AUDIO_CHANNELS.
    device : int | str | None
        sounddevice device index or name. None = system default.
        Defaults to config.AUDIO_DEVICE_INDEX.
    """

    def __init__(
        self,
        callback: ChunkCallback,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        chunk_duration: float = config.AUDIO_CHUNK_DURATION,
        channels: int = config.AUDIO_CHANNELS,
        device: Optional[int | str] = config.AUDIO_DEVICE_INDEX,
    ) -> None:
        self._callback = callback
        self._sample_rate = sample_rate
        self._chunk_samples = int(sample_rate * chunk_duration)
        self._channels = channels
        self._device = device

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()

        # Buffer accumulates raw samples between sounddevice callbacks
        # until we have a full chunk.
        self._buffer = np.empty((0, channels), dtype=np.float32)

        logger.debug(
            "AudioRecorder initialised: device=%s sample_rate=%d "
            "chunk_samples=%d channels=%d",
            device,
            sample_rate,
            self._chunk_samples,
            channels,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """
        Open the input stream and begin delivering chunks to the callback.

        Raises AudioRecorderError if the stream cannot be opened.
        """
        with self._lock:
            if self._running:
                logger.warning("AudioRecorder.start() called while already running.")
                return

            try:
                self._stream = sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype="float32",
                    device=self._device,
                    callback=self._sounddevice_callback,
                    # blocksize=0 lets sounddevice choose; we accumulate
                    # in _buffer and emit complete chunks ourselves.
                    blocksize=0,
                )
                self._stream.start()
                self._running = True
                logger.info(
                    "AudioRecorder started (device=%s, %d Hz, %d ch).",
                    self._device,
                    self._sample_rate,
                    self._channels,
                )
            except sd.PortAudioError as exc:
                raise AudioRecorderError(
                    f"Failed to open audio input stream: {exc}"
                ) from exc

    def stop(self) -> None:
        """Stop the input stream and release hardware resources."""
        with self._lock:
            if not self._running:
                return
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
                    self._stream = None
            except sd.PortAudioError as exc:
                logger.error("Error stopping audio stream: %s", exc)
            finally:
                self._running = False
                self._buffer = np.empty((0, self._channels), dtype=np.float32)
                logger.info("AudioRecorder stopped.")

    # ------------------------------------------------------------------
    # Internal sounddevice callback
    # ------------------------------------------------------------------

    def _sounddevice_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Called by sounddevice on its own thread each time new samples arrive.

        We accumulate samples in self._buffer and fire our callback
        whenever we have a full chunk.
        """
        if status:
            logger.warning("AudioRecorder stream status: %s", status)

        # indata shape is (frames, channels); append to buffer
        self._buffer = np.concatenate([self._buffer, indata], axis=0)

        # Emit as many complete chunks as the buffer holds
        while len(self._buffer) >= self._chunk_samples:
            chunk = self._buffer[: self._chunk_samples]
            self._buffer = self._buffer[self._chunk_samples :]

            # Squeeze to 1-D for mono (makes downstream code simpler)
            if self._channels == 1:
                chunk = chunk[:, 0]

            try:
                self._callback(chunk, self._sample_rate)
            except Exception as exc:  # noqa: BLE001
                logger.error("Exception in AudioRecorder callback: %s", exc)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "AudioRecorder":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Utility: list available input devices
# ---------------------------------------------------------------------------

def list_input_devices() -> list[dict]:
    """
    Return a list of available audio input devices.

    Each entry is a dict with keys:
        index       int   — device index for use in config / AudioRecorder
        name        str   — human-readable name
        channels    int   — maximum input channels
        sample_rate float — default sample rate
        is_default  bool  — True for the system default input device
    """
    try:
        default_index = sd.default.device[0]  # input device index
    except Exception:
        default_index = -1

    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                    "is_default": idx == default_index,
                }
            )
    return devices


def print_input_devices() -> None:
    """Print available input devices to stdout in a readable format."""
    devices = list_input_devices()
    if not devices:
        print("No input devices found.")
        return

    print(f"\n{'IDX':>4}  {'DEFAULT':>7}  {'RATE':>7}  {'CH':>3}  NAME")
    print("-" * 70)
    for dev in devices:
        default_marker = "  *    " if dev["is_default"] else "       "
        print(
            f"{dev['index']:>4}  {default_marker}  "
            f"{int(dev['sample_rate']):>7}  {dev['channels']:>3}  {dev['name']}"
        )
    print()


if __name__ == "__main__":
    print_input_devices()
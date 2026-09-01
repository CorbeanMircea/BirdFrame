"""
Tests for AudioRecorder.

We never open a real microphone in automated tests.
Instead we drive the recorder's internal callback directly,
which is the same code path used during real recording.
"""

import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.audio.recorder import AudioRecorder, AudioRecorderError, list_input_devices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_indata(frames: int, channels: int = 1) -> np.ndarray:
    """Create a fake sounddevice indata array (frames, channels)."""
    return np.random.randn(frames, channels).astype(np.float32)


def _silent_callback(chunk: np.ndarray, sample_rate: int) -> None:
    """No-op chunk callback."""
    pass


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

class TestAudioRecorderInit:
    def test_default_construction(self):
        recorder = AudioRecorder(callback=_silent_callback)
        assert recorder.sample_rate > 0
        assert not recorder.is_running

    def test_custom_sample_rate(self):
        recorder = AudioRecorder(callback=_silent_callback, sample_rate=16000)
        assert recorder.sample_rate == 16000

    def test_not_running_before_start(self):
        recorder = AudioRecorder(callback=_silent_callback)
        assert recorder.is_running is False


# ---------------------------------------------------------------------------
# Chunk delivery tests (no real microphone)
# ---------------------------------------------------------------------------

class TestChunkDelivery:
    """
    Feed synthetic audio directly into _sounddevice_callback and verify
    that the user callback receives correctly sized chunks.
    """

    def _make_recorder(self, callback, sample_rate=16000, chunk_duration=1.0, channels=1):
        return AudioRecorder(
            callback=callback,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            channels=channels,
        )

    def test_single_exact_chunk(self):
        """One block of exactly chunk_samples frames → exactly one callback."""
        received = []
        recorder = self._make_recorder(lambda c, sr: received.append(c))

        chunk_samples = recorder._chunk_samples
        indata = _make_indata(chunk_samples, channels=1)
        recorder._sounddevice_callback(indata, chunk_samples, None, MagicMock())

        assert len(received) == 1
        assert received[0].shape == (chunk_samples,)

    def test_two_exact_chunks(self):
        """Two blocks delivered sequentially → two callbacks."""
        received = []
        recorder = self._make_recorder(lambda c, sr: received.append(c))

        chunk_samples = recorder._chunk_samples
        for _ in range(2):
            indata = _make_indata(chunk_samples)
            recorder._sounddevice_callback(indata, chunk_samples, None, MagicMock())

        assert len(received) == 2

    def test_partial_block_not_delivered(self):
        """Half a chunk's worth of data → no callback yet."""
        received = []
        recorder = self._make_recorder(lambda c, sr: received.append(c))

        half = recorder._chunk_samples // 2
        indata = _make_indata(half)
        recorder._sounddevice_callback(indata, half, None, MagicMock())

        assert len(received) == 0

    def test_partial_then_complete(self):
        """Half block + half block → exactly one callback."""
        received = []
        recorder = self._make_recorder(lambda c, sr: received.append(c))

        half = recorder._chunk_samples // 2
        recorder._sounddevice_callback(_make_indata(half), half, None, MagicMock())
        recorder._sounddevice_callback(_make_indata(half), half, None, MagicMock())

        assert len(received) == 1

    def test_oversized_block_yields_multiple_chunks(self):
        """3× chunk_samples in one block → three callbacks."""
        received = []
        recorder = self._make_recorder(lambda c, sr: received.append(c))

        n = recorder._chunk_samples * 3
        indata = _make_indata(n)
        recorder._sounddevice_callback(indata, n, None, MagicMock())

        assert len(received) == 3

    def test_chunk_values_match_input(self):
        """The delivered chunk contains the same samples we fed in."""
        received = []
        recorder = self._make_recorder(lambda c, sr: received.append(c.copy()))

        chunk_samples = recorder._chunk_samples
        indata = _make_indata(chunk_samples)
        recorder._sounddevice_callback(indata, chunk_samples, None, MagicMock())

        np.testing.assert_array_almost_equal(received[0], indata[:, 0])

    def test_sample_rate_passed_to_callback(self):
        """Callback receives the correct sample rate."""
        rates = []
        recorder = self._make_recorder(
            lambda c, sr: rates.append(sr), sample_rate=22050
        )
        chunk_samples = recorder._chunk_samples
        recorder._sounddevice_callback(
            _make_indata(chunk_samples), chunk_samples, None, MagicMock()
        )
        assert rates[0] == 22050

    def test_mono_chunk_is_1d(self):
        """Mono chunks must be squeezed to 1-D arrays."""
        received = []
        recorder = self._make_recorder(
            lambda c, sr: received.append(c), channels=1
        )
        chunk_samples = recorder._chunk_samples
        recorder._sounddevice_callback(
            _make_indata(chunk_samples, channels=1), chunk_samples, None, MagicMock()
        )
        assert received[0].ndim == 1

    def test_callback_exception_does_not_crash_recorder(self):
        """An exception inside the user callback must not propagate."""
        def bad_callback(chunk, sr):
            raise RuntimeError("simulated callback error")

        recorder = self._make_recorder(bad_callback)
        chunk_samples = recorder._chunk_samples
        # Must not raise
        recorder._sounddevice_callback(
            _make_indata(chunk_samples), chunk_samples, None, MagicMock()
        )

    def test_remainder_carried_to_next_block(self):
        """Leftover samples after a full chunk are kept for the next call."""
        received = []
        recorder = self._make_recorder(lambda c, sr: received.append(c))

        chunk_samples = recorder._chunk_samples
        # Send 1.5 chunks worth of data
        n = int(chunk_samples * 1.5)
        recorder._sounddevice_callback(_make_indata(n), n, None, MagicMock())

        # Should deliver exactly one chunk
        assert len(received) == 1
        # Buffer should hold the remaining 0.5 chunk
        assert len(recorder._buffer) == n - chunk_samples


# ---------------------------------------------------------------------------
# Start / stop tests (mocked sounddevice)
# ---------------------------------------------------------------------------

class TestStartStop:
    def _patched_recorder(self):
        """Return a recorder with sounddevice.InputStream mocked out."""
        mock_stream = MagicMock()
        with patch("backend.audio.recorder.sd.InputStream", return_value=mock_stream):
            recorder = AudioRecorder(callback=_silent_callback)
            return recorder, mock_stream

    def test_start_sets_running(self):
        mock_stream = MagicMock()
        with patch("backend.audio.recorder.sd.InputStream", return_value=mock_stream):
            recorder = AudioRecorder(callback=_silent_callback)
            recorder.start()
            assert recorder.is_running is True
            recorder.stop()

    def test_stop_clears_running(self):
        mock_stream = MagicMock()
        with patch("backend.audio.recorder.sd.InputStream", return_value=mock_stream):
            recorder = AudioRecorder(callback=_silent_callback)
            recorder.start()
            recorder.stop()
            assert recorder.is_running is False

    def test_double_start_is_safe(self):
        mock_stream = MagicMock()
        with patch("backend.audio.recorder.sd.InputStream", return_value=mock_stream):
            recorder = AudioRecorder(callback=_silent_callback)
            recorder.start()
            recorder.start()  # second call must not raise
            assert recorder.is_running is True
            recorder.stop()

    def test_stop_without_start_is_safe(self):
        recorder = AudioRecorder(callback=_silent_callback)
        recorder.stop()  # must not raise
        assert recorder.is_running is False

    def test_context_manager(self):
        mock_stream = MagicMock()
        with patch("backend.audio.recorder.sd.InputStream", return_value=mock_stream):
            with AudioRecorder(callback=_silent_callback) as recorder:
                assert recorder.is_running is True
            assert recorder.is_running is False

    def test_portaudio_error_raises_recorder_error(self):
        import sounddevice as sd
        with patch(
            "backend.audio.recorder.sd.InputStream",
            side_effect=sd.PortAudioError("no device"),
        ):
            recorder = AudioRecorder(callback=_silent_callback)
            with pytest.raises(AudioRecorderError):
                recorder.start()


# ---------------------------------------------------------------------------
# list_input_devices tests (mocked sounddevice)
# ---------------------------------------------------------------------------

class TestListInputDevices:
    def _fake_devices(self):
        return [
            {"name": "Microphone (HD Audio)", "max_input_channels": 2,
             "max_output_channels": 0, "default_samplerate": 48000.0},
            {"name": "Speakers (HD Audio)",   "max_input_channels": 0,
             "max_output_channels": 2, "default_samplerate": 48000.0},
            {"name": "Stereo Mix",            "max_input_channels": 2,
             "max_output_channels": 0, "default_samplerate": 44100.0},
        ]

    def test_only_input_devices_returned(self):
        with patch("backend.audio.recorder.sd.query_devices", return_value=self._fake_devices()), \
             patch("backend.audio.recorder.sd.default") as mock_default:
            mock_default.device = [0, 1]
            devices = list_input_devices()
        names = [d["name"] for d in devices]
        assert "Speakers (HD Audio)" not in names
        assert "Microphone (HD Audio)" in names

    def test_device_dict_has_required_keys(self):
        with patch("backend.audio.recorder.sd.query_devices", return_value=self._fake_devices()), \
             patch("backend.audio.recorder.sd.default") as mock_default:
            mock_default.device = [0, 1]
            devices = list_input_devices()
        for dev in devices:
            assert "index" in dev
            assert "name" in dev
            assert "channels" in dev
            assert "sample_rate" in dev
            assert "is_default" in dev

    def test_default_device_flagged(self):
        with patch("backend.audio.recorder.sd.query_devices", return_value=self._fake_devices()), \
             patch("backend.audio.recorder.sd.default") as mock_default:
            mock_default.device = [0, 1]
            devices = list_input_devices()
        defaults = [d for d in devices if d["is_default"]]
        assert len(defaults) == 1
        assert defaults[0]["index"] == 0
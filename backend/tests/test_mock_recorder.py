"""
Tests for MockAudioRecorder and the WAV loading / writing utilities.

Also contains an end-to-end pipeline test:
    MockAudioRecorder → AudioProcessor → chunk collection

No real microphone is used anywhere in this file.
"""

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.audio.mock_recorder import (
    MockAudioRecorder,
    MockAudioRecorderError,
    _load_wav,
    write_wav,
)
from backend.audio.processor import AudioProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


def _sine(seconds: float, freq: float = 440.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _silence(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# WAV write / load round-trip tests
# ---------------------------------------------------------------------------

class TestWavRoundTrip:
    def test_write_and_load(self, tmp_path):
        path = tmp_path / "test.wav"
        original = _sine(1.0)
        write_wav(path, original, SAMPLE_RATE)

        loaded, sr = _load_wav(path)
        assert sr == SAMPLE_RATE
        assert len(loaded) == len(original)
        # 16-bit quantisation introduces small errors; tolerate ~0.001
        np.testing.assert_allclose(loaded, original, atol=1e-3)

    def test_loaded_dtype_is_float32(self, tmp_path):
        path = tmp_path / "dtype.wav"
        write_wav(path, _silence(0.5), SAMPLE_RATE)
        loaded, _ = _load_wav(path)
        assert loaded.dtype == np.float32

    def test_loaded_values_in_range(self, tmp_path):
        path = tmp_path / "range.wav"
        write_wav(path, _sine(0.5), SAMPLE_RATE)
        loaded, _ = _load_wav(path)
        assert loaded.min() >= -1.0
        assert loaded.max() <= 1.0

    def test_load_missing_file_raises(self):
        with pytest.raises(MockAudioRecorderError, match="not found"):
            _load_wav(Path("/nonexistent/path/audio.wav"))

    def test_stereo_wav_mixed_to_mono(self, tmp_path):
        """A 2-channel WAV must be mixed down to 1-D mono."""
        import wave
        path = tmp_path / "stereo.wav"
        n_frames = SAMPLE_RATE  # 1 second
        pcm_left  = (np.ones(n_frames) * 0.5 * 32767).astype(np.int16)
        pcm_right = (np.ones(n_frames) * 0.5 * 32767).astype(np.int16)
        interleaved = np.empty(n_frames * 2, dtype=np.int16)
        interleaved[0::2] = pcm_left
        interleaved[1::2] = pcm_right

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(interleaved.tobytes())

        loaded, sr = _load_wav(path)
        assert loaded.ndim == 1
        assert sr == SAMPLE_RATE


# ---------------------------------------------------------------------------
# MockAudioRecorder.from_array tests
# ---------------------------------------------------------------------------

class TestFromArray:
    def test_delivers_all_chunks(self):
        audio = _silence(3.0)  # 3 seconds
        chunks = []
        recorder = MockAudioRecorder.from_array(
            audio, SAMPLE_RATE,
            callback=lambda c, sr: chunks.append(c),
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=5.0)
        assert len(chunks) == 3

    def test_chunk_sample_rate_correct(self):
        rates = []
        recorder = MockAudioRecorder.from_array(
            _silence(1.0), SAMPLE_RATE,
            callback=lambda c, sr: rates.append(sr),
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=5.0)
        assert all(r == SAMPLE_RATE for r in rates)

    def test_chunk_length_correct(self):
        lengths = []
        recorder = MockAudioRecorder.from_array(
            _silence(2.0), SAMPLE_RATE,
            callback=lambda c, sr: lengths.append(len(c)),
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=5.0)
        expected = int(1.0 * SAMPLE_RATE)
        assert all(l == expected for l in lengths)

    def test_not_running_after_join(self):
        recorder = MockAudioRecorder.from_array(
            _silence(1.0), SAMPLE_RATE,
            callback=lambda c, sr: None,
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=5.0)
        assert recorder.is_running is False

    def test_2d_audio_raises(self):
        with pytest.raises(MockAudioRecorderError, match="1-D"):
            MockAudioRecorder(
                audio=np.zeros((100, 2), dtype=np.float32),
                sample_rate=SAMPLE_RATE,
                callback=lambda c, sr: None,
            )

    def test_double_start_is_safe(self):
        recorder = MockAudioRecorder.from_array(
            _silence(2.0), SAMPLE_RATE,
            callback=lambda c, sr: None,
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.start()  # must not raise or create second thread
        recorder.join(timeout=5.0)

    def test_context_manager(self):
        chunks = []
        with MockAudioRecorder.from_array(
            _silence(1.0), SAMPLE_RATE,
            callback=lambda c, sr: chunks.append(c),
            chunk_duration=1.0,
        ) as recorder:
            recorder.join(timeout=5.0)
        assert recorder.is_running is False

    def test_stop_before_complete(self):
        """stop() must not hang even if playback is not finished."""
        recorder = MockAudioRecorder.from_array(
            _silence(60.0), SAMPLE_RATE,   # very long audio
            callback=lambda c, sr: time.sleep(0.01),
            chunk_duration=1.0,
            realtime=False,
        )
        recorder.start()
        time.sleep(0.05)
        recorder.stop()   # must return quickly
        assert recorder.is_running is False


# ---------------------------------------------------------------------------
# MockAudioRecorder.from_wav tests
# ---------------------------------------------------------------------------

class TestFromWav:
    def test_loads_and_delivers(self, tmp_path):
        path = tmp_path / "bird.wav"
        write_wav(path, _sine(2.0), SAMPLE_RATE)

        chunks = []
        recorder = MockAudioRecorder.from_wav(
            path,
            callback=lambda c, sr: chunks.append(c),
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=5.0)
        assert len(chunks) == 2

    def test_missing_wav_raises(self):
        with pytest.raises(MockAudioRecorderError, match="not found"):
            MockAudioRecorder.from_wav(
                "/nonexistent/bird.wav",
                callback=lambda c, sr: None,
            )


# ---------------------------------------------------------------------------
# End-to-end pipeline test: MockAudioRecorder → AudioProcessor
# ---------------------------------------------------------------------------

class TestMockRecorderToProcessor:
    """
    Wire MockAudioRecorder directly into AudioProcessor and verify that
    segments come out the other end with correct length and count.

    This is the integration smoke-test for the audio input stage.
    """

    def test_segments_received(self):
        """3 seconds of audio with 1s segments, 0s overlap → 3 segments."""
        segments = []

        processor = AudioProcessor(
            segment_callback=lambda seg, sr: segments.append((seg, sr)),
            segment_duration=1.0,
            overlap_duration=0.0,
        )

        recorder = MockAudioRecorder.from_array(
            _sine(3.0), SAMPLE_RATE,
            callback=processor.process,
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=10.0)

        assert len(segments) == 3

    def test_segment_sample_rate_correct(self):
        segments = []
        processor = AudioProcessor(
            segment_callback=lambda seg, sr: segments.append((seg, sr)),
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        recorder = MockAudioRecorder.from_array(
            _silence(2.0), SAMPLE_RATE,
            callback=processor.process,
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=10.0)
        assert all(sr == SAMPLE_RATE for _, sr in segments)

    def test_segment_length_correct(self):
        segments = []
        processor = AudioProcessor(
            segment_callback=lambda seg, sr: segments.append((seg, sr)),
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        recorder = MockAudioRecorder.from_array(
            _silence(2.0), SAMPLE_RATE,
            callback=processor.process,
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=10.0)
        expected = int(1.0 * SAMPLE_RATE)
        assert all(len(seg) == expected for seg, _ in segments)

    def test_overlap_pipeline(self):
        """1s segments with 0.5s overlap over 2s audio → 3 segments."""
        segments = []
        processor = AudioProcessor(
            segment_callback=lambda seg, sr: segments.append(seg),
            segment_duration=1.0,
            overlap_duration=0.5,
        )
        recorder = MockAudioRecorder.from_array(
            _sine(2.0), SAMPLE_RATE,
            callback=processor.process,
            chunk_duration=0.5,   # recorder chunks smaller than processor segment
        )
        recorder.start()
        recorder.join(timeout=10.0)
        assert len(segments) == 3

    def test_wav_file_pipeline(self, tmp_path):
        """WAV file → MockAudioRecorder → AudioProcessor → segments."""
        path = tmp_path / "pipeline.wav"
        write_wav(path, _sine(3.0), SAMPLE_RATE)

        segments = []
        processor = AudioProcessor(
            segment_callback=lambda seg, sr: segments.append(seg),
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        recorder = MockAudioRecorder.from_wav(
            path,
            callback=processor.process,
            chunk_duration=1.0,
        )
        recorder.start()
        recorder.join(timeout=10.0)
        assert len(segments) == 3
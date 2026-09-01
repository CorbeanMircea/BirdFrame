"""
Tests for AudioProcessor.

No microphone or real audio stream required — we feed synthetic numpy
arrays directly into processor.process().
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.audio.processor import AudioProcessor, AudioProcessorError, _resample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # use a small rate so tests run fast


def _make_chunk(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a chunk of silence of the given length."""
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def _make_tone(seconds: float, freq: float = 440.0,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a sine-wave chunk."""
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _collect_processor(
    segment_duration: float = 1.0,
    overlap_duration: float = 0.0,
    target_sample_rate=None,
) -> tuple[AudioProcessor, list]:
    """Return a processor and a list that accumulates emitted segments."""
    segments = []
    processor = AudioProcessor(
        segment_callback=lambda seg, sr: segments.append((seg.copy(), sr)),
        segment_duration=segment_duration,
        overlap_duration=overlap_duration,
        target_sample_rate=target_sample_rate,
    )
    return processor, segments


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_valid_construction(self):
        proc, _ = _collect_processor(segment_duration=3.0, overlap_duration=0.5)
        assert proc.sample_rate is None       # not yet seen any data
        assert proc.segment_samples is None

    def test_overlap_equal_to_segment_raises(self):
        with pytest.raises(AudioProcessorError, match="overlap"):
            AudioProcessor(
                segment_callback=lambda s, r: None,
                segment_duration=1.0,
                overlap_duration=1.0,
            )

    def test_overlap_greater_than_segment_raises(self):
        with pytest.raises(AudioProcessorError, match="overlap"):
            AudioProcessor(
                segment_callback=lambda s, r: None,
                segment_duration=1.0,
                overlap_duration=2.0,
            )


# ---------------------------------------------------------------------------
# Basic segment emission tests
# ---------------------------------------------------------------------------

class TestSegmentEmission:
    def test_exact_one_segment(self):
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        proc.process(_make_chunk(1.0), SAMPLE_RATE)
        assert len(segs) == 1

    def test_less_than_one_segment_no_emission(self):
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        proc.process(_make_chunk(0.5), SAMPLE_RATE)
        assert len(segs) == 0

    def test_two_exact_segments(self):
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        proc.process(_make_chunk(2.0), SAMPLE_RATE)
        assert len(segs) == 2

    def test_accumulated_across_calls(self):
        """Two half-second chunks → one one-second segment."""
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        proc.process(_make_chunk(0.5), SAMPLE_RATE)
        assert len(segs) == 0
        proc.process(_make_chunk(0.5), SAMPLE_RATE)
        assert len(segs) == 1

    def test_segment_length_correct(self):
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        proc.process(_make_chunk(1.0), SAMPLE_RATE)
        expected = int(1.0 * SAMPLE_RATE)
        assert len(segs[0][0]) == expected

    def test_sample_rate_passed_to_callback(self):
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        proc.process(_make_chunk(1.0), SAMPLE_RATE)
        assert segs[0][1] == SAMPLE_RATE

    def test_segment_values_preserved(self):
        """Samples must pass through unchanged when no resampling."""
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        tone = _make_tone(1.0)
        proc.process(tone, SAMPLE_RATE)
        np.testing.assert_array_almost_equal(segs[0][0], tone)


# ---------------------------------------------------------------------------
# Overlap tests
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_overlap_produces_more_segments(self):
        """
        With 50 % overlap, 2 seconds of audio should produce 3 segments
        (at t=0, t=0.5, t=1.0) rather than 2.
        """
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.5)
        proc.process(_make_chunk(2.0), SAMPLE_RATE)
        assert len(segs) == 3

    def test_overlap_segment_shares_tail_of_previous(self):
        """
        The first 'overlap' samples of segment N+1 should equal the
        last 'overlap' samples of segment N.
        """
        overlap = 0.5
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=overlap)
        tone = _make_tone(3.0)
        proc.process(tone, SAMPLE_RATE)

        overlap_samples = int(overlap * SAMPLE_RATE)
        # Last overlap_samples of segment 0
        tail_of_first = segs[0][0][-overlap_samples:]
        # First overlap_samples of segment 1
        head_of_second = segs[1][0][:overlap_samples]
        np.testing.assert_array_almost_equal(tail_of_first, head_of_second)

    def test_zero_overlap_no_shared_samples(self):
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        proc.process(_make_chunk(2.0), SAMPLE_RATE)
        # Each segment should start immediately after the previous ends
        assert len(segs) == 2
        # No overlap means segments are independent slices
        seg_len = int(1.0 * SAMPLE_RATE)
        assert len(segs[0][0]) == seg_len
        assert len(segs[1][0]) == seg_len


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_non_array_raises(self):
        proc, _ = _collect_processor()
        with pytest.raises(AudioProcessorError, match="ndarray"):
            proc.process([0.0, 1.0, 2.0], SAMPLE_RATE)  # type: ignore

    def test_2d_array_raises(self):
        proc, _ = _collect_processor()
        with pytest.raises(AudioProcessorError, match="1-D"):
            proc.process(np.zeros((100, 2), dtype=np.float32), SAMPLE_RATE)

    def test_sample_rate_change_mid_stream_raises(self):
        proc, _ = _collect_processor()
        proc.process(_make_chunk(0.1), SAMPLE_RATE)
        with pytest.raises(AudioProcessorError, match="Sample rate changed"):
            proc.process(_make_chunk(0.1), SAMPLE_RATE + 1)

    def test_int16_input_converted_to_float32(self):
        """int16 arrays should be silently cast to float32."""
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        chunk = np.zeros(SAMPLE_RATE, dtype=np.int16)
        proc.process(chunk, SAMPLE_RATE)
        assert segs[0][0].dtype == np.float32


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_buffer(self):
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        proc.process(_make_chunk(0.5), SAMPLE_RATE)  # half a segment
        proc.reset()
        proc.process(_make_chunk(0.5), SAMPLE_RATE)  # another half
        # If reset worked, the two halves are NOT joined → no segment yet
        assert len(segs) == 0

    def test_reset_clears_sample_rate(self):
        proc, _ = _collect_processor()
        proc.process(_make_chunk(0.1), SAMPLE_RATE)
        assert proc.sample_rate == SAMPLE_RATE
        proc.reset()
        assert proc.sample_rate is None

    def test_reset_allows_new_sample_rate(self):
        proc, segs = _collect_processor(segment_duration=1.0, overlap_duration=0.0)
        proc.process(_make_chunk(0.5), SAMPLE_RATE)
        proc.reset()
        new_rate = 8000
        proc.process(_make_chunk(1.0, sample_rate=new_rate), new_rate)
        assert len(segs) == 1
        assert segs[0][1] == new_rate


# ---------------------------------------------------------------------------
# Resampling tests
# ---------------------------------------------------------------------------

class TestResampling:
    def test_output_rate_matches_target(self):
        proc, segs = _collect_processor(
            segment_duration=1.0,
            overlap_duration=0.0,
            target_sample_rate=8000,
        )
        proc.process(_make_chunk(1.0, sample_rate=SAMPLE_RATE), SAMPLE_RATE)
        assert segs[0][1] == 8000

    def test_output_length_matches_target_rate(self):
        target = 8000
        proc, segs = _collect_processor(
            segment_duration=1.0,
            overlap_duration=0.0,
            target_sample_rate=target,
        )
        proc.process(_make_chunk(1.0, sample_rate=SAMPLE_RATE), SAMPLE_RATE)
        expected_len = int(1.0 * target)
        assert len(segs[0][0]) == expected_len

    def test_no_resampling_when_rates_match(self):
        proc, segs = _collect_processor(
            segment_duration=1.0,
            overlap_duration=0.0,
            target_sample_rate=SAMPLE_RATE,  # same as input
        )
        proc.process(_make_chunk(1.0), SAMPLE_RATE)
        assert len(segs[0][0]) == SAMPLE_RATE

    def test_resample_helper_upsample(self):
        audio = _make_tone(1.0, sample_rate=8000)
        result = _resample(audio, from_rate=8000, to_rate=16000)
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_resample_helper_downsample(self):
        audio = _make_tone(1.0, sample_rate=16000)
        result = _resample(audio, from_rate=16000, to_rate=8000)
        assert len(result) == 8000

    def test_resample_helper_same_rate_passthrough(self):
        audio = _make_tone(1.0)
        result = _resample(audio, from_rate=SAMPLE_RATE, to_rate=SAMPLE_RATE)
        np.testing.assert_array_equal(result, audio)

    def test_callback_exception_does_not_propagate(self):
        def bad_callback(seg, sr):
            raise RuntimeError("downstream failure")

        proc = AudioProcessor(
            segment_callback=bad_callback,
            segment_duration=1.0,
            overlap_duration=0.0,
        )
        # Must not raise even though callback raises
        proc.process(_make_chunk(1.0), SAMPLE_RATE)
"""
Tests for BirdDetector.

Uses synthetic audio (sine waves, silence, noise) to exercise every
code path without requiring a real microphone or bird recordings.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.audio.detector import (
    BirdDetector,
    BirdDetectorError,
    DetectorResult,
    _compute_rms,
    _compute_band_energy_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


def _silence(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _sine(freq: float, seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    """White noise across all frequencies."""
    rng = np.random.default_rng(seed=42)
    return (amplitude * rng.uniform(-1.0, 1.0, int(seconds * SAMPLE_RATE))).astype(np.float32)


def _low_freq_noise(seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    """Sine wave at 200 Hz — below the bird frequency band."""
    return _sine(200.0, seconds, amplitude)


def _bird_freq_tone(seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    """Sine wave at 4 000 Hz — squarely inside the bird frequency band."""
    return _sine(4000.0, seconds, amplitude)


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_construction(self):
        detector = BirdDetector()
        assert detector.energy_threshold >= 0
        assert detector.freq_min < detector.freq_max
        assert 0.0 <= detector.band_ratio_threshold <= 1.0

    def test_custom_parameters(self):
        detector = BirdDetector(
            energy_threshold=0.01,
            freq_min=2000,
            freq_max=8000,
            band_ratio_threshold=0.1,
        )
        assert detector.energy_threshold == pytest.approx(0.01)
        assert detector.freq_min == 2000
        assert detector.freq_max == 8000

    def test_freq_min_equal_max_raises(self):
        with pytest.raises(BirdDetectorError, match="freq_min"):
            BirdDetector(freq_min=5000, freq_max=5000)

    def test_freq_min_greater_than_max_raises(self):
        with pytest.raises(BirdDetectorError, match="freq_min"):
            BirdDetector(freq_min=8000, freq_max=1000)

    def test_negative_energy_threshold_raises(self):
        with pytest.raises(BirdDetectorError, match="energy_threshold"):
            BirdDetector(energy_threshold=-0.1)

    def test_band_ratio_out_of_range_raises(self):
        with pytest.raises(BirdDetectorError, match="band_ratio_threshold"):
            BirdDetector(band_ratio_threshold=1.5)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_non_array_raises(self):
        detector = BirdDetector()
        with pytest.raises(BirdDetectorError, match="ndarray"):
            detector.is_bird_audio([0.0, 1.0], SAMPLE_RATE)  # type: ignore

    def test_2d_array_raises(self):
        detector = BirdDetector()
        with pytest.raises(BirdDetectorError, match="1-D"):
            detector.is_bird_audio(np.zeros((100, 2), dtype=np.float32), SAMPLE_RATE)

    def test_empty_array_raises(self):
        detector = BirdDetector()
        with pytest.raises(BirdDetectorError, match="empty"):
            detector.is_bird_audio(np.array([], dtype=np.float32), SAMPLE_RATE)

    def test_zero_sample_rate_raises(self):
        detector = BirdDetector()
        with pytest.raises(BirdDetectorError, match="sample_rate"):
            detector.is_bird_audio(_silence(), sample_rate=0)

    def test_negative_sample_rate_raises(self):
        detector = BirdDetector()
        with pytest.raises(BirdDetectorError, match="sample_rate"):
            detector.is_bird_audio(_silence(), sample_rate=-1)


# ---------------------------------------------------------------------------
# Energy threshold tests
# ---------------------------------------------------------------------------

class TestEnergyThreshold:
    def test_silence_rejected(self):
        detector = BirdDetector(energy_threshold=0.001)
        assert detector.is_bird_audio(_silence(), SAMPLE_RATE) is False

    def test_loud_signal_not_rejected_by_energy(self):
        """A loud bird-frequency tone must pass the energy check."""
        detector = BirdDetector(energy_threshold=0.001, band_ratio_threshold=0.0)
        assert detector.is_bird_audio(_bird_freq_tone(amplitude=0.8), SAMPLE_RATE) is True

    def test_very_quiet_signal_rejected(self):
        detector = BirdDetector(energy_threshold=0.1)
        quiet = _bird_freq_tone(amplitude=0.001)
        assert detector.is_bird_audio(quiet, SAMPLE_RATE) is False

    def test_zero_threshold_accepts_any_nonsilent(self):
        """energy_threshold=0 means even near-silent audio passes energy check."""
        detector = BirdDetector(energy_threshold=0.0, band_ratio_threshold=0.0)
        assert detector.is_bird_audio(_bird_freq_tone(amplitude=0.001), SAMPLE_RATE) is True

    def test_rejection_reason_mentions_rms(self):
        detector = BirdDetector(energy_threshold=0.5)
        result = detector.analyse(_silence(), SAMPLE_RATE)
        assert result.accepted is False
        assert "RMS" in result.rejection_reason


# ---------------------------------------------------------------------------
# Band energy tests
# ---------------------------------------------------------------------------

class TestBandEnergy:
    def test_bird_frequency_tone_accepted(self):
        """4 000 Hz tone is inside the default bird band → accepted."""
        detector = BirdDetector(
            energy_threshold=0.001,
            freq_min=1000,
            freq_max=10000,
            band_ratio_threshold=0.05,
        )
        assert detector.is_bird_audio(_bird_freq_tone(), SAMPLE_RATE) is True

    def test_low_frequency_tone_rejected(self):
        """200 Hz tone is outside the bird band → rejected."""
        detector = BirdDetector(
            energy_threshold=0.001,
            freq_min=1000,
            freq_max=10000,
            band_ratio_threshold=0.5,  # strict: require 50 % in band
        )
        assert detector.is_bird_audio(_low_freq_noise(), SAMPLE_RATE) is False

    def test_zero_band_ratio_threshold_accepts_any_loud_signal(self):
        """With band_ratio_threshold=0, any loud signal passes."""
        detector = BirdDetector(energy_threshold=0.001, band_ratio_threshold=0.0)
        assert detector.is_bird_audio(_low_freq_noise(amplitude=0.5), SAMPLE_RATE) is True

    def test_rejection_reason_mentions_band(self):
        detector = BirdDetector(
            energy_threshold=0.001,
            freq_min=1000,
            freq_max=10000,
            band_ratio_threshold=0.9,  # impossibly strict
        )
        result = detector.analyse(_low_freq_noise(), SAMPLE_RATE)
        assert result.accepted is False
        assert "Band" in result.rejection_reason

    def test_white_noise_has_energy_across_all_bands(self):
        """White noise should have some energy in the bird band."""
        detector = BirdDetector(
            energy_threshold=0.001,
            freq_min=1000,
            freq_max=10000,
            band_ratio_threshold=0.05,
        )
        assert detector.is_bird_audio(_noise(), SAMPLE_RATE) is True


# ---------------------------------------------------------------------------
# DetectorResult tests
# ---------------------------------------------------------------------------

class TestDetectorResult:
    def test_result_fields_populated(self):
        detector = BirdDetector(energy_threshold=0.001, band_ratio_threshold=0.05)
        result = detector.analyse(_bird_freq_tone(), SAMPLE_RATE)
        assert isinstance(result, DetectorResult)
        assert isinstance(result.rms_energy, float)
        assert isinstance(result.band_energy_ratio, float)
        assert result.rms_threshold == pytest.approx(0.001)
        assert result.band_ratio_threshold == pytest.approx(0.05)

    def test_accepted_result_has_no_rejection_reason(self):
        detector = BirdDetector(energy_threshold=0.001, band_ratio_threshold=0.0)
        result = detector.analyse(_bird_freq_tone(), SAMPLE_RATE)
        assert result.accepted is True
        assert result.rejection_reason is None

    def test_rejected_result_has_rejection_reason(self):
        detector = BirdDetector(energy_threshold=0.001)
        result = detector.analyse(_silence(), SAMPLE_RATE)
        assert result.accepted is False
        assert result.rejection_reason is not None
        assert len(result.rejection_reason) > 0

    def test_rms_energy_positive_for_nonsilent(self):
        detector = BirdDetector()
        result = detector.analyse(_bird_freq_tone(), SAMPLE_RATE)
        assert result.rms_energy > 0.0

    def test_rms_energy_zero_for_silence(self):
        detector = BirdDetector()
        result = detector.analyse(_silence(), SAMPLE_RATE)
        assert result.rms_energy == pytest.approx(0.0)

    def test_band_ratio_between_zero_and_one(self):
        detector = BirdDetector()
        result = detector.analyse(_bird_freq_tone(), SAMPLE_RATE)
        assert 0.0 <= result.band_energy_ratio <= 1.0


# ---------------------------------------------------------------------------
# Signal utility tests
# ---------------------------------------------------------------------------

class TestSignalUtilities:
    def test_rms_silence_is_zero(self):
        assert _compute_rms(_silence()) == pytest.approx(0.0)

    def test_rms_sine_correct(self):
        # RMS of a sine wave with amplitude A is A / sqrt(2)
        amplitude = 0.6
        tone = _sine(1000.0, amplitude=amplitude)
        expected = amplitude / np.sqrt(2)
        assert _compute_rms(tone) == pytest.approx(expected, rel=0.01)

    def test_band_ratio_pure_tone_in_band(self):
        """A 4 kHz tone should have nearly all energy in 1k–10k band."""
        tone = _sine(4000.0)
        ratio = _compute_band_energy_ratio(tone, SAMPLE_RATE, 1000, 10000)
        assert ratio > 0.95

    def test_band_ratio_pure_tone_out_of_band(self):
        """A 200 Hz tone should have nearly zero energy in 1k–10k band."""
        tone = _sine(200.0)
        ratio = _compute_band_energy_ratio(tone, SAMPLE_RATE, 1000, 10000)
        assert ratio < 0.05

    def test_band_ratio_silence_is_zero(self):
        ratio = _compute_band_energy_ratio(_silence(), SAMPLE_RATE, 1000, 10000)
        assert ratio == pytest.approx(0.0)

    def test_band_ratio_in_range(self):
        ratio = _compute_band_energy_ratio(_noise(), SAMPLE_RATE, 1000, 10000)
        assert 0.0 <= ratio <= 1.0
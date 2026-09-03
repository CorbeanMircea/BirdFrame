"""
Tests for BirdNetIdentifier.

Split into two groups:

1. Unit tests — never load the real model. Use mocks to test the
   identifier's own logic (validation, result mapping, error handling).
   These run in CI with no GPU/model required.

2. Integration test — loads the real BirdNET model and runs inference
   on a synthetic audio segment. Marked with @pytest.mark.integration
   and skipped by default. Run manually with:
       pytest -m integration backend/tests/test_birdnet_identifier.py -v
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.identification.birdnet_identifier import (
    BirdNetIdentifier,
    BIRDNET_SAMPLE_RATE,
    RecordingBuffer,
)
from backend.identification.base import BirdIdentifierError, IdentificationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = BIRDNET_SAMPLE_RATE  # 48 000


def _segment(seconds=3.0, sr=SR) -> np.ndarray:
    """Sine wave at 4 kHz — non-silent, plausible bird frequency."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)


def _warmed_identifier(**kwargs) -> BirdNetIdentifier:
    """Return a BirdNetIdentifier with a mock analyzer already loaded."""
    identifier = BirdNetIdentifier(**kwargs)
    identifier._analyzer = MagicMock()
    identifier._available = True
    return identifier


def _make_mock_recording(detections):
    """Return a mock RecordingBuffer with pre-set detections."""
    mock = MagicMock()
    mock.detections = detections
    return mock


FAKE_DETECTIONS = [
    {
        "scientific_name": "Erithacus rubecula",
        "common_name": "European Robin",
        "confidence": 0.87,
        "start_time": 0.0,
        "end_time": 3.0,
    },
    {
        "scientific_name": "Parus major",
        "common_name": "Great Tit",
        "confidence": 0.65,
        "start_time": 0.0,
        "end_time": 3.0,
    },
    {
        "scientific_name": "Turdus merula",
        "common_name": "Eurasian Blackbird",
        "confidence": 0.42,
        "start_time": 0.0,
        "end_time": 3.0,
    },
]


# ---------------------------------------------------------------------------
# Helper: run identify() with mocked RecordingBuffer
# ---------------------------------------------------------------------------

def _identify_with_fake_detections(identifier, detections, top_n=5):
    """Patch RecordingBuffer at module level and run identify()."""
    mock_recording = _make_mock_recording(detections)
    with patch(
        "backend.identification.birdnet_identifier.RecordingBuffer",
        return_value=mock_recording,
    ):
        return identifier.identify(_segment(), SR, top_n=top_n)


# ---------------------------------------------------------------------------
# Unit tests (no real model)
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_construction(self):
        identifier = BirdNetIdentifier()
        assert identifier.model_name == "BirdNET"
        assert identifier.model_version == "2.4"
        assert identifier.is_available is False  # not warmed up yet

    def test_custom_location(self):
        identifier = BirdNetIdentifier(latitude=45.9, longitude=24.9)
        assert identifier._latitude == pytest.approx(45.9)
        assert identifier._longitude == pytest.approx(24.9)

    def test_invalid_min_confidence_raises(self):
        with pytest.raises(BirdIdentifierError, match="min_confidence"):
            BirdNetIdentifier(min_confidence=1.5)

    def test_repr_contains_model_name(self):
        identifier = BirdNetIdentifier()
        assert "BirdNET" in repr(identifier)


class TestWarmup:
    def test_warmup_sets_available(self):
        identifier = BirdNetIdentifier()
        mock_analyzer = MagicMock()
        with patch(
            "backend.identification.birdnet_identifier._BirdnetAnalyzer",
            return_value=mock_analyzer,
        ):
            identifier.warmup()
        assert identifier._available is True
        assert identifier._analyzer is mock_analyzer

    def test_double_warmup_is_noop(self):
        """Calling warmup() twice must not reload the model."""
        identifier = BirdNetIdentifier()
        mock_analyzer = MagicMock()
        with patch(
            "backend.identification.birdnet_identifier._BirdnetAnalyzer",
            return_value=mock_analyzer,
        ) as mock_cls:
            identifier.warmup()
            identifier.warmup()  # second call
            assert mock_cls.call_count == 1  # only instantiated once

    def test_identify_without_warmup_raises(self):
        identifier = BirdNetIdentifier()
        with pytest.raises(BirdIdentifierError, match="warmup"):
            identifier.identify(_segment(), SR)

    def test_warmup_unavailable_when_birdnetlib_missing(self):
        identifier = BirdNetIdentifier()
        with patch(
            "backend.identification.birdnet_identifier._BIRDNETLIB_AVAILABLE",
            False,
        ):
            with pytest.raises(BirdIdentifierError, match="birdnetlib"):
                identifier.warmup()


class TestInputValidation:
    def test_non_array_raises(self):
        identifier = _warmed_identifier()
        with pytest.raises(BirdIdentifierError, match="ndarray"):
            identifier.identify([0.0, 1.0], SR)  # type: ignore

    def test_2d_array_raises(self):
        identifier = _warmed_identifier()
        with pytest.raises(BirdIdentifierError, match="1-D"):
            identifier.identify(np.zeros((100, 2), dtype=np.float32), SR)

    def test_empty_segment_raises(self):
        identifier = _warmed_identifier()
        with pytest.raises(BirdIdentifierError, match="empty"):
            identifier.identify(np.array([], dtype=np.float32), SR)

    def test_zero_sample_rate_raises(self):
        identifier = _warmed_identifier()
        with pytest.raises(BirdIdentifierError, match="sample_rate"):
            identifier.identify(_segment(), sample_rate=0)


class TestResultMapping:
    def test_returns_identification_results(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS
        )
        assert all(isinstance(r, IdentificationResult) for r in results)

    def test_scientific_name_mapped(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS
        )
        names = [r.scientific_name for r in results]
        assert "Erithacus rubecula" in names

    def test_common_name_mapped(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS
        )
        names = [r.common_name for r in results]
        assert "European Robin" in names

    def test_confidence_mapped(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS
        )
        robin = next(r for r in results if r.scientific_name == "Erithacus rubecula")
        assert robin.confidence == pytest.approx(0.87)

    def test_model_name_on_result(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS
        )
        assert all(r.model_name == "BirdNET" for r in results)

    def test_model_version_on_result(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS
        )
        assert all(r.model_version == "2.4" for r in results)

    def test_duration_populated(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS
        )
        assert all(r.duration_seconds is not None for r in results)
        assert all(r.duration_seconds > 0 for r in results)

    def test_sorted_descending(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS
        )
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_empty_detections_returns_empty_list(self):
        results = _identify_with_fake_detections(_warmed_identifier(), [])
        assert results == []

    def test_top_n_respected(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS, top_n=2
        )
        assert len(results) <= 2

    def test_raw_score_populated(self):
        results = _identify_with_fake_detections(
            _warmed_identifier(), FAKE_DETECTIONS
        )
        assert all(r.raw_score is not None for r in results)

    def test_malformed_detection_skipped(self):
        """A detection missing required keys must be skipped, not crash."""
        bad = [{"scientific_name": "Erithacus rubecula"}]  # missing fields
        results = _identify_with_fake_detections(_warmed_identifier(), bad)
        assert results == []


class TestLocationFiltering:
    def test_location_kwargs_passed_to_recording(self):
        identifier = _warmed_identifier(latitude=45.9, longitude=24.9, week=24)
        mock_recording = _make_mock_recording([])
        with patch(
            "backend.identification.birdnet_identifier.RecordingBuffer",
            return_value=mock_recording,
        ) as mock_buffer:
            identifier.identify(_segment(), SR)

        call_kwargs = mock_buffer.call_args.kwargs
        assert call_kwargs.get("lat") == pytest.approx(45.9)
        assert call_kwargs.get("lon") == pytest.approx(24.9)
        assert call_kwargs.get("week_48") == 24

    def test_no_location_no_lat_lon_kwargs(self):
        identifier = _warmed_identifier()  # no lat/lon
        mock_recording = _make_mock_recording([])
        with patch(
            "backend.identification.birdnet_identifier.RecordingBuffer",
            return_value=mock_recording,
        ) as mock_buffer:
            identifier.identify(_segment(), SR)

        call_kwargs = mock_buffer.call_args.kwargs
        assert "lat" not in call_kwargs
        assert "lon" not in call_kwargs

    def test_buffer_kwarg_passed(self):
        """Confirm 'buffer' (not 'data') is the parameter used."""
        identifier = _warmed_identifier()
        mock_recording = _make_mock_recording([])
        with patch(
            "backend.identification.birdnet_identifier.RecordingBuffer",
            return_value=mock_recording,
        ) as mock_buffer:
            seg = _segment()
            identifier.identify(seg, SR)

        call_kwargs = mock_buffer.call_args.kwargs
        assert "buffer" in call_kwargs

    def test_rate_kwarg_passed(self):
        """Confirm sample rate is passed as 'rate'."""
        identifier = _warmed_identifier()
        mock_recording = _make_mock_recording([])
        with patch(
            "backend.identification.birdnet_identifier.RecordingBuffer",
            return_value=mock_recording,
        ) as mock_buffer:
            identifier.identify(_segment(), SR)

        call_kwargs = mock_buffer.call_args.kwargs
        assert call_kwargs.get("rate") == SR


# ---------------------------------------------------------------------------
# Integration test (loads real BirdNET model — skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBirdNetIntegration:
    """
    Runs real BirdNET inference. Requires birdnetlib + tensorflow installed
    and model weights downloaded.

    Run with:
        pytest -m integration backend/tests/test_birdnet_identifier.py -v
    """

    @pytest.fixture(scope="class")
    def identifier(self):
        """Load the real BirdNET model once for all integration tests."""
        ident = BirdNetIdentifier(
            latitude=45.9,   # Romania
            longitude=24.9,
            week=24,         # June
        )
        ident.warmup()
        return ident

    def test_is_available_after_warmup(self, identifier):
        assert identifier.is_available is True

    def test_identify_returns_list(self, identifier):
        results = identifier.identify(_segment(seconds=3.0), SR)
        assert isinstance(results, list)

    def test_results_are_identification_results(self, identifier):
        results = identifier.identify(_segment(seconds=3.0), SR)
        assert all(isinstance(r, IdentificationResult) for r in results)

    def test_confidences_in_range(self, identifier):
        results = identifier.identify(_segment(seconds=3.0), SR)
        assert all(0.0 <= r.confidence <= 1.0 for r in results)

    def test_results_sorted_descending(self, identifier):
        results = identifier.identify(_segment(seconds=3.0), SR)
        if len(results) > 1:
            confidences = [r.confidence for r in results]
            assert confidences == sorted(confidences, reverse=True)

    def test_second_warmup_is_noop(self, identifier):
        identifier.warmup()
        assert identifier.is_available is True
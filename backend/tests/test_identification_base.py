"""
Tests for the BirdIdentifier base class and IdentificationResult type.

Uses a minimal concrete subclass to test the abstract interface without
requiring any real model.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.identification.base import (
    BirdIdentifier,
    BirdIdentifierError,
    IdentificationResult,
)


# ---------------------------------------------------------------------------
# Minimal concrete implementation for testing the base class
# ---------------------------------------------------------------------------

class _AlwaysRobinIdentifier(BirdIdentifier):
    """Returns a single fixed result regardless of input."""

    @property
    def model_name(self) -> str:
        return "AlwaysRobin"

    @property
    def model_version(self) -> str:
        return "0.0"

    def identify(
        self,
        segment: np.ndarray,
        sample_rate: int,
        top_n: int = 5,
    ) -> list[IdentificationResult]:
        return [
            IdentificationResult(
                scientific_name="Erithacus rubecula",
                common_name="European Robin",
                confidence=0.95,
                model_name=self.model_name,
                model_version=self.model_version,
                duration_seconds=len(segment) / sample_rate,
            )
        ]


class _EmptyIdentifier(BirdIdentifier):
    """Always returns an empty list (no detections)."""

    def identify(self, segment, sample_rate, top_n=5):
        return []


class _MultiResultIdentifier(BirdIdentifier):
    """Returns multiple candidates sorted by confidence descending."""

    def identify(self, segment, sample_rate, top_n=5):
        candidates = [
            IdentificationResult("Parus major", "Great Tit", 0.90,
                                 "Multi", "1.0"),
            IdentificationResult("Erithacus rubecula", "European Robin", 0.75,
                                 "Multi", "1.0"),
            IdentificationResult("Turdus merula", "Eurasian Blackbird", 0.55,
                                 "Multi", "1.0"),
            IdentificationResult("Passer domesticus", "House Sparrow", 0.30,
                                 "Multi", "1.0"),
        ]
        return sorted(candidates[:top_n], key=lambda r: r.confidence, reverse=True)


# ---------------------------------------------------------------------------
# IdentificationResult tests
# ---------------------------------------------------------------------------

class TestIdentificationResult:
    def test_basic_construction(self):
        r = IdentificationResult(
            scientific_name="Erithacus rubecula",
            common_name="European Robin",
            confidence=0.87,
            model_name="TestModel",
            model_version="1.0",
        )
        assert r.scientific_name == "Erithacus rubecula"
        assert r.common_name == "European Robin"
        assert r.confidence == pytest.approx(0.87)
        assert r.model_name == "TestModel"
        assert r.duration_seconds is None
        assert r.raw_score is None

    def test_timestamp_defaults_to_utc_now(self):
        before = datetime.now(timezone.utc)
        r = IdentificationResult("A b", "Common A", 0.5, "M", "1")
        after = datetime.now(timezone.utc)
        assert before <= r.timestamp <= after

    def test_confidence_zero_is_valid(self):
        r = IdentificationResult("A b", "Common A", 0.0, "M", "1")
        assert r.confidence == 0.0

    def test_confidence_one_is_valid(self):
        r = IdentificationResult("A b", "Common A", 1.0, "M", "1")
        assert r.confidence == 1.0

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            IdentificationResult("A b", "Common A", 1.01, "M", "1")

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            IdentificationResult("A b", "Common A", -0.01, "M", "1")

    def test_empty_scientific_name_raises(self):
        with pytest.raises(ValueError, match="scientific_name"):
            IdentificationResult("", "Common A", 0.5, "M", "1")

    def test_whitespace_scientific_name_raises(self):
        with pytest.raises(ValueError, match="scientific_name"):
            IdentificationResult("   ", "Common A", 0.5, "M", "1")

    def test_empty_common_name_raises(self):
        with pytest.raises(ValueError, match="common_name"):
            IdentificationResult("A b", "", 0.5, "M", "1")

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="duration_seconds"):
            IdentificationResult("A b", "Common A", 0.5, "M", "1",
                                 duration_seconds=-1.0)

    def test_zero_duration_raises(self):
        with pytest.raises(ValueError, match="duration_seconds"):
            IdentificationResult("A b", "Common A", 0.5, "M", "1",
                                 duration_seconds=0.0)

    def test_positive_duration_valid(self):
        r = IdentificationResult("A b", "Common A", 0.5, "M", "1",
                                 duration_seconds=3.0)
        assert r.duration_seconds == pytest.approx(3.0)

    def test_raw_score_stored(self):
        r = IdentificationResult("A b", "Common A", 0.5, "M", "1",
                                 raw_score=0.123)
        assert r.raw_score == pytest.approx(0.123)

    def test_above_threshold_true(self):
        r = IdentificationResult("A b", "Common A", 0.8, "M", "1")
        assert r.above_threshold(0.5) is True

    def test_above_threshold_false(self):
        r = IdentificationResult("A b", "Common A", 0.3, "M", "1")
        assert r.above_threshold(0.5) is False

    def test_above_threshold_equal(self):
        r = IdentificationResult("A b", "Common A", 0.5, "M", "1")
        assert r.above_threshold(0.5) is True

    def test_repr_contains_species(self):
        r = IdentificationResult("Erithacus rubecula", "European Robin",
                                 0.9, "M", "1")
        assert "Erithacus rubecula" in repr(r)
        assert "0.90" in repr(r)


# ---------------------------------------------------------------------------
# BirdIdentifier abstract class tests
# ---------------------------------------------------------------------------

class TestBirdIdentifierInterface:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BirdIdentifier()  # type: ignore

    def test_concrete_subclass_instantiates(self):
        identifier = _AlwaysRobinIdentifier()
        assert identifier is not None

    def test_model_name_property(self):
        identifier = _AlwaysRobinIdentifier()
        assert identifier.model_name == "AlwaysRobin"

    def test_model_version_property(self):
        identifier = _AlwaysRobinIdentifier()
        assert identifier.model_version == "0.0"

    def test_is_available_default_true(self):
        identifier = _AlwaysRobinIdentifier()
        assert identifier.is_available is True

    def test_warmup_does_not_raise(self):
        identifier = _AlwaysRobinIdentifier()
        identifier.warmup()  # default implementation — must not raise

    def test_repr_contains_model_name(self):
        identifier = _AlwaysRobinIdentifier()
        assert "AlwaysRobin" in repr(identifier)


# ---------------------------------------------------------------------------
# identify() contract tests
# ---------------------------------------------------------------------------

class TestIdentifyContract:
    def test_returns_list(self):
        identifier = _AlwaysRobinIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        result = identifier.identify(segment, sample_rate=16000)
        assert isinstance(result, list)

    def test_returns_identification_results(self):
        identifier = _AlwaysRobinIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        results = identifier.identify(segment, sample_rate=16000)
        assert all(isinstance(r, IdentificationResult) for r in results)

    def test_empty_result_is_valid(self):
        identifier = _EmptyIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        results = identifier.identify(segment, sample_rate=16000)
        assert results == []

    def test_results_sorted_descending(self):
        identifier = _MultiResultIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        results = identifier.identify(segment, sample_rate=16000)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_top_n_respected(self):
        identifier = _MultiResultIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        results = identifier.identify(segment, sample_rate=16000, top_n=2)
        assert len(results) <= 2

    def test_duration_populated(self):
        identifier = _AlwaysRobinIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        results = identifier.identify(segment, sample_rate=16000)
        assert results[0].duration_seconds == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# identify_and_filter() tests
# ---------------------------------------------------------------------------

class TestIdentifyAndFilter:
    def test_filters_below_threshold(self):
        identifier = _MultiResultIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        results = identifier.identify_and_filter(
            segment, sample_rate=16000, min_confidence=0.8
        )
        assert all(r.confidence >= 0.8 for r in results)

    def test_empty_when_all_below_threshold(self):
        identifier = _MultiResultIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        results = identifier.identify_and_filter(
            segment, sample_rate=16000, min_confidence=0.99
        )
        assert results == []

    def test_all_pass_when_threshold_zero(self):
        identifier = _MultiResultIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        all_results = identifier.identify(segment, sample_rate=16000)
        filtered = identifier.identify_and_filter(
            segment, sample_rate=16000, min_confidence=0.0
        )
        assert len(filtered) == len(all_results)

    def test_still_sorted_after_filter(self):
        identifier = _MultiResultIdentifier()
        segment = np.zeros(16000, dtype=np.float32)
        results = identifier.identify_and_filter(
            segment, sample_rate=16000, min_confidence=0.5
        )
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)
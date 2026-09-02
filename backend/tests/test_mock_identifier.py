"""
Tests for MockBirdIdentifier.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.identification.mock_identifier import MockBirdIdentifier, _DEFAULT_SPECIES
from backend.identification.base import BirdIdentifierError, IdentificationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


def _segment(seconds: float = 3.0) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_construction(self):
        m = MockBirdIdentifier()
        assert m.model_name == "MockBirdIdentifier"
        assert m.model_version == "0.1"
        assert m.is_available is True

    def test_invalid_mode_raises(self):
        with pytest.raises(BirdIdentifierError, match="mode"):
            MockBirdIdentifier(mode="turbo")  # type: ignore

    def test_invalid_fixed_confidence_raises(self):
        with pytest.raises(BirdIdentifierError, match="fixed_confidence"):
            MockBirdIdentifier(fixed_confidence=1.5)

    def test_invalid_confidence_range_raises(self):
        with pytest.raises(BirdIdentifierError, match="confidence_range"):
            MockBirdIdentifier(confidence_range=(0.9, 0.1))

    def test_negative_delay_raises(self):
        with pytest.raises(BirdIdentifierError, match="call_delay"):
            MockBirdIdentifier(call_delay=-1.0)

    def test_empty_species_list_raises(self):
        with pytest.raises(BirdIdentifierError, match="empty"):
            MockBirdIdentifier(species_list=[])

    def test_warmup_does_not_raise(self):
        MockBirdIdentifier().warmup()

    def test_call_count_starts_at_zero(self):
        assert MockBirdIdentifier().call_count == 0


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_non_array_raises(self):
        m = MockBirdIdentifier(mode="fixed")
        with pytest.raises(BirdIdentifierError, match="ndarray"):
            m.identify([0.0, 1.0], SAMPLE_RATE)  # type: ignore

    def test_2d_array_raises(self):
        m = MockBirdIdentifier(mode="fixed")
        with pytest.raises(BirdIdentifierError, match="1-D"):
            m.identify(np.zeros((100, 2), dtype=np.float32), SAMPLE_RATE)

    def test_empty_segment_raises(self):
        m = MockBirdIdentifier(mode="fixed")
        with pytest.raises(BirdIdentifierError, match="empty"):
            m.identify(np.array([], dtype=np.float32), SAMPLE_RATE)

    def test_zero_sample_rate_raises(self):
        m = MockBirdIdentifier(mode="fixed")
        with pytest.raises(BirdIdentifierError, match="sample_rate"):
            m.identify(_segment(), sample_rate=0)


# ---------------------------------------------------------------------------
# Empty mode tests
# ---------------------------------------------------------------------------

class TestEmptyMode:
    def test_returns_empty_list(self):
        m = MockBirdIdentifier(mode="empty")
        assert m.identify(_segment(), SAMPLE_RATE) == []

    def test_call_count_increments(self):
        m = MockBirdIdentifier(mode="empty")
        m.identify(_segment(), SAMPLE_RATE)
        m.identify(_segment(), SAMPLE_RATE)
        assert m.call_count == 2


# ---------------------------------------------------------------------------
# Fixed mode tests
# ---------------------------------------------------------------------------

class TestFixedMode:
    def test_returns_identification_results(self):
        m = MockBirdIdentifier(mode="fixed")
        results = m.identify(_segment(), SAMPLE_RATE)
        assert all(isinstance(r, IdentificationResult) for r in results)

    def test_first_species_is_european_robin(self):
        m = MockBirdIdentifier(mode="fixed")
        results = m.identify(_segment(), SAMPLE_RATE)
        assert results[0].scientific_name == "Erithacus rubecula"

    def test_results_sorted_descending(self):
        m = MockBirdIdentifier(mode="fixed")
        results = m.identify(_segment(), SAMPLE_RATE)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_top_n_respected(self):
        m = MockBirdIdentifier(mode="fixed")
        results = m.identify(_segment(), SAMPLE_RATE, top_n=3)
        assert len(results) <= 3

    def test_same_result_every_call(self):
        m = MockBirdIdentifier(mode="fixed")
        r1 = m.identify(_segment(), SAMPLE_RATE)
        r2 = m.identify(_segment(), SAMPLE_RATE)
        assert [r.scientific_name for r in r1] == [r.scientific_name for r in r2]

    def test_confidence_matches_fixed_value(self):
        m = MockBirdIdentifier(mode="fixed", fixed_confidence=0.77)
        results = m.identify(_segment(), SAMPLE_RATE, top_n=1)
        assert results[0].confidence == pytest.approx(0.77)

    def test_duration_seconds_populated(self):
        m = MockBirdIdentifier(mode="fixed")
        results = m.identify(_segment(seconds=3.0), SAMPLE_RATE)
        assert results[0].duration_seconds == pytest.approx(3.0)

    def test_model_name_on_result(self):
        m = MockBirdIdentifier(mode="fixed")
        results = m.identify(_segment(), SAMPLE_RATE, top_n=1)
        assert results[0].model_name == "MockBirdIdentifier"


# ---------------------------------------------------------------------------
# Sequential mode tests
# ---------------------------------------------------------------------------

class TestSequentialMode:
    def test_returns_one_result_per_call(self):
        m = MockBirdIdentifier(mode="sequential")
        results = m.identify(_segment(), SAMPLE_RATE)
        assert len(results) == 1

    def test_cycles_through_species(self):
        m = MockBirdIdentifier(mode="sequential")
        names = [
            m.identify(_segment(), SAMPLE_RATE)[0].scientific_name
            for _ in range(len(_DEFAULT_SPECIES) + 1)
        ]
        # After exhausting the list it wraps around
        assert names[0] == names[len(_DEFAULT_SPECIES)]

    def test_sequential_advances_each_call(self):
        m = MockBirdIdentifier(mode="sequential")
        first = m.identify(_segment(), SAMPLE_RATE)[0].scientific_name
        second = m.identify(_segment(), SAMPLE_RATE)[0].scientific_name
        assert first != second

    def test_reset_restarts_sequence(self):
        m = MockBirdIdentifier(mode="sequential")
        first_run = m.identify(_segment(), SAMPLE_RATE)[0].scientific_name
        m.reset()
        after_reset = m.identify(_segment(), SAMPLE_RATE)[0].scientific_name
        assert first_run == after_reset


# ---------------------------------------------------------------------------
# Random mode tests
# ---------------------------------------------------------------------------

class TestRandomMode:
    def test_returns_results(self):
        m = MockBirdIdentifier(mode="random", seed=42)
        results = m.identify(_segment(), SAMPLE_RATE)
        assert len(results) > 0

    def test_confidences_within_range(self):
        m = MockBirdIdentifier(
            mode="random", seed=0,
            confidence_range=(0.6, 0.9),
        )
        for _ in range(10):
            results = m.identify(_segment(), SAMPLE_RATE)
            for r in results:
                assert 0.6 <= r.confidence <= 0.9

    def test_results_sorted_descending(self):
        m = MockBirdIdentifier(mode="random", seed=1)
        results = m.identify(_segment(), SAMPLE_RATE)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_seed_produces_deterministic_output(self):
        m1 = MockBirdIdentifier(mode="random", seed=99)
        m2 = MockBirdIdentifier(mode="random", seed=99)
        r1 = [r.scientific_name for r in m1.identify(_segment(), SAMPLE_RATE)]
        r2 = [r.scientific_name for r in m2.identify(_segment(), SAMPLE_RATE)]
        assert r1 == r2

    def test_different_seeds_different_output(self):
        m1 = MockBirdIdentifier(mode="random", seed=1)
        m2 = MockBirdIdentifier(mode="random", seed=2)
        r1 = [r.scientific_name for r in m1.identify(_segment(), SAMPLE_RATE)]
        r2 = [r.scientific_name for r in m2.identify(_segment(), SAMPLE_RATE)]
        assert r1 != r2

    def test_top_n_respected(self):
        m = MockBirdIdentifier(mode="random", seed=7)
        results = m.identify(_segment(), SAMPLE_RATE, top_n=2)
        assert len(results) <= 2

    def test_species_from_list(self):
        custom = [("Anas platyrhynchos", "Mallard"),
                  ("Ardea cinerea", "Grey Heron")]
        m = MockBirdIdentifier(mode="random", species_list=custom, seed=0)
        results = m.identify(_segment(), SAMPLE_RATE)
        valid_names = {s[0] for s in custom}
        assert all(r.scientific_name in valid_names for r in results)


# ---------------------------------------------------------------------------
# identify_and_filter integration tests
# ---------------------------------------------------------------------------

class TestIdentifyAndFilter:
    def test_filter_applied(self):
        m = MockBirdIdentifier(mode="fixed", fixed_confidence=0.85)
        results = m.identify_and_filter(_segment(), SAMPLE_RATE,
                                        min_confidence=0.90)
        assert all(r.confidence >= 0.90 for r in results)

    def test_filter_empty_when_threshold_high(self):
        m = MockBirdIdentifier(mode="fixed", fixed_confidence=0.5)
        results = m.identify_and_filter(_segment(), SAMPLE_RATE,
                                        min_confidence=0.99)
        assert results == []


# ---------------------------------------------------------------------------
# Call count and reset tests
# ---------------------------------------------------------------------------

class TestCallCountAndReset:
    def test_call_count_increments(self):
        m = MockBirdIdentifier(mode="fixed")
        m.identify(_segment(), SAMPLE_RATE)
        m.identify(_segment(), SAMPLE_RATE)
        assert m.call_count == 2

    def test_reset_clears_call_count(self):
        m = MockBirdIdentifier(mode="fixed")
        m.identify(_segment(), SAMPLE_RATE)
        m.reset()
        assert m.call_count == 0

    def test_repr_contains_model_name(self):
        m = MockBirdIdentifier()
        assert "MockBirdIdentifier" in repr(m)
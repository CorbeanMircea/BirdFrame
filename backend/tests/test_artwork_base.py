"""
Tests for ArtworkProvider base class.

Uses minimal concrete subclasses to test the abstract interface and
the default concrete methods (get_artwork_for_species_list, missing_artwork).
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.artwork.base import ArtworkProvider, ArtworkProviderError


# ---------------------------------------------------------------------------
# Minimal concrete implementations for testing
# ---------------------------------------------------------------------------

class _AlwaysHasArtwork(ArtworkProvider):
    """Returns a fake path for every species."""

    def __init__(self, fake_path: Path):
        self._path = fake_path

    def get_artwork(self, scientific_name: str) -> Path:
        return self._path

    def has_artwork(self, scientific_name: str) -> bool:
        return True


class _NeverHasArtwork(ArtworkProvider):
    """Returns None for every species."""

    def get_artwork(self, scientific_name: str) -> None:
        return None

    def has_artwork(self, scientific_name: str) -> bool:
        return False


class _SelectiveArtwork(ArtworkProvider):
    """Has artwork only for species in a fixed set."""

    KNOWN = {"Erithacus rubecula", "Parus major"}

    def get_artwork(self, scientific_name: str) -> Path | None:
        if scientific_name in self.KNOWN:
            return Path(f"/fake/{scientific_name}.jpg")
        return None

    def has_artwork(self, scientific_name: str) -> bool:
        return scientific_name in self.KNOWN

    @property
    def supported_species(self) -> list[str]:
        return list(self.KNOWN)


class _ErrorProvider(ArtworkProvider):
    """Always raises ArtworkProviderError."""

    def get_artwork(self, scientific_name: str) -> None:
        raise ArtworkProviderError("Simulated provider error")

    def has_artwork(self, scientific_name: str) -> bool:
        raise ArtworkProviderError("Simulated provider error")


# ---------------------------------------------------------------------------
# Abstract class tests
# ---------------------------------------------------------------------------

class TestAbstractClass:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ArtworkProvider()  # type: ignore

    def test_concrete_subclass_instantiates(self, tmp_path):
        fake = tmp_path / "robin.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        assert provider is not None

    def test_provider_name_default(self, tmp_path):
        fake = tmp_path / "robin.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        assert provider.provider_name == "_AlwaysHasArtwork"

    def test_supported_species_default_empty(self, tmp_path):
        fake = tmp_path / "robin.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        assert provider.supported_species == []

    def test_supported_species_override(self):
        provider = _SelectiveArtwork()
        assert "Erithacus rubecula" in provider.supported_species

    def test_repr_contains_class_name(self, tmp_path):
        fake = tmp_path / "robin.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        assert "_AlwaysHasArtwork" in repr(provider)


# ---------------------------------------------------------------------------
# get_artwork() tests
# ---------------------------------------------------------------------------

class TestGetArtwork:
    def test_returns_path_when_available(self, tmp_path):
        fake = tmp_path / "robin.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        result = provider.get_artwork("Erithacus rubecula")
        assert result == fake

    def test_returns_none_when_unavailable(self):
        provider = _NeverHasArtwork()
        result = provider.get_artwork("Erithacus rubecula")
        assert result is None

    def test_selective_returns_path_for_known(self):
        provider = _SelectiveArtwork()
        result = provider.get_artwork("Erithacus rubecula")
        assert result is not None

    def test_selective_returns_none_for_unknown(self):
        provider = _SelectiveArtwork()
        result = provider.get_artwork("Unknown birdus")
        assert result is None


# ---------------------------------------------------------------------------
# has_artwork() tests
# ---------------------------------------------------------------------------

class TestHasArtwork:
    def test_true_when_available(self, tmp_path):
        fake = tmp_path / "robin.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        assert provider.has_artwork("Erithacus rubecula") is True

    def test_false_when_unavailable(self):
        provider = _NeverHasArtwork()
        assert provider.has_artwork("Erithacus rubecula") is False

    def test_selective_true_for_known(self):
        provider = _SelectiveArtwork()
        assert provider.has_artwork("Erithacus rubecula") is True

    def test_selective_false_for_unknown(self):
        provider = _SelectiveArtwork()
        assert provider.has_artwork("Unknown birdus") is False


# ---------------------------------------------------------------------------
# get_artwork_for_species_list() tests
# ---------------------------------------------------------------------------

class TestGetArtworkForSpeciesList:
    def test_returns_dict(self, tmp_path):
        fake = tmp_path / "bird.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        result = provider.get_artwork_for_species_list(["Erithacus rubecula"])
        assert isinstance(result, dict)

    def test_keys_match_input(self, tmp_path):
        fake = tmp_path / "bird.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        names = ["Erithacus rubecula", "Parus major"]
        result = provider.get_artwork_for_species_list(names)
        assert set(result.keys()) == set(names)

    def test_none_for_missing(self):
        provider = _NeverHasArtwork()
        result = provider.get_artwork_for_species_list(["Erithacus rubecula"])
        assert result["Erithacus rubecula"] is None

    def test_mixed_results(self):
        provider = _SelectiveArtwork()
        names = ["Erithacus rubecula", "Unknown birdus"]
        result = provider.get_artwork_for_species_list(names)
        assert result["Erithacus rubecula"] is not None
        assert result["Unknown birdus"] is None

    def test_error_in_provider_returns_none_not_raises(self):
        """ArtworkProviderError must be caught, not propagated."""
        provider = _ErrorProvider()
        result = provider.get_artwork_for_species_list(["Erithacus rubecula"])
        assert result["Erithacus rubecula"] is None

    def test_empty_list_returns_empty_dict(self, tmp_path):
        fake = tmp_path / "bird.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        result = provider.get_artwork_for_species_list([])
        assert result == {}


# ---------------------------------------------------------------------------
# missing_artwork() tests
# ---------------------------------------------------------------------------

class TestMissingArtwork:
    def test_returns_all_when_none_available(self):
        provider = _NeverHasArtwork()
        names = ["Erithacus rubecula", "Parus major"]
        missing = provider.missing_artwork(names)
        assert set(missing) == set(names)

    def test_returns_empty_when_all_available(self, tmp_path):
        fake = tmp_path / "bird.jpg"
        fake.touch()
        provider = _AlwaysHasArtwork(fake)
        names = ["Erithacus rubecula", "Parus major"]
        missing = provider.missing_artwork(names)
        assert missing == []

    def test_returns_only_missing(self):
        provider = _SelectiveArtwork()
        names = ["Erithacus rubecula", "Unknown birdus", "Parus major"]
        missing = provider.missing_artwork(names)
        assert "Unknown birdus" in missing
        assert "Erithacus rubecula" not in missing
        assert "Parus major" not in missing

    def test_empty_input_returns_empty(self):
        provider = _NeverHasArtwork()
        assert provider.missing_artwork([]) == []
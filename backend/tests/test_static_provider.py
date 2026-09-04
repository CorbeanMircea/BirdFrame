"""
Tests for StaticArtworkProvider.

Uses a temporary directory as the assets directory so no real
artwork files are needed for the test suite.
"""

import sys
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.artwork.static_provider import (
    StaticArtworkProvider,
    scientific_name_to_filename,
    SUPPORTED_EXTENSIONS,
)
from backend.artwork.base import ArtworkProviderError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_artwork(directory: Path, scientific_name: str,
                       ext: str = ".jpg") -> Path:
    """Create an empty image file with the correct filename."""
    from backend.artwork.static_provider import scientific_name_to_filename
    stem = scientific_name_to_filename(scientific_name)
    path = directory / f"{stem}{ext}"
    path.write_bytes(b"fake image data")
    return path


# ---------------------------------------------------------------------------
# scientific_name_to_filename tests
# ---------------------------------------------------------------------------

class TestScientificNameToFilename:
    def test_basic_conversion(self):
        assert scientific_name_to_filename("Erithacus rubecula") == \
               "erithacus_rubecula"

    def test_lowercase(self):
        assert scientific_name_to_filename("Parus Major") == "parus_major"

    def test_already_lowercase(self):
        assert scientific_name_to_filename("parus major") == "parus_major"

    def test_three_word_name(self):
        # Subspecies names
        result = scientific_name_to_filename("Parus major major")
        assert result == "parus_major_major"

    def test_strips_whitespace(self):
        assert scientific_name_to_filename("  Parus major  ") == "parus_major"

    def test_removes_special_characters(self):
        result = scientific_name_to_filename("Parus (major)")
        # Parentheses removed, spaces → underscores
        assert "(" not in result
        assert ")" not in result

    def test_empty_string(self):
        assert scientific_name_to_filename("") == ""


# ---------------------------------------------------------------------------
# StaticArtworkProvider tests
# ---------------------------------------------------------------------------

class TestStaticArtworkProvider:
    def test_default_construction(self, tmp_path):
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider is not None
        assert provider.provider_name == "StaticArtworkProvider"

    def test_repr(self, tmp_path):
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert "StaticArtworkProvider" in repr(provider)

    # ------------------------------------------------------------------
    # get_artwork() tests
    # ------------------------------------------------------------------

    def test_returns_path_for_existing_jpg(self, tmp_path):
        path = _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        result = provider.get_artwork("Erithacus rubecula")
        assert result == path

    def test_returns_path_for_existing_png(self, tmp_path):
        path = _make_fake_artwork(tmp_path, "Parus major", ".png")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        result = provider.get_artwork("Parus major")
        assert result == path

    def test_returns_path_for_existing_webp(self, tmp_path):
        path = _make_fake_artwork(tmp_path, "Turdus merula", ".webp")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        result = provider.get_artwork("Turdus merula")
        assert result == path

    def test_returns_none_for_missing(self, tmp_path):
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        result = provider.get_artwork("Unknown birdus")
        assert result is None

    def test_case_insensitive_lookup(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        # Lookup with different case
        result = provider.get_artwork("erithacus rubecula")
        assert result is not None

    def test_empty_name_raises(self, tmp_path):
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        with pytest.raises(ArtworkProviderError):
            provider.get_artwork("")

    def test_whitespace_name_raises(self, tmp_path):
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        with pytest.raises(ArtworkProviderError):
            provider.get_artwork("   ")

    def test_missing_directory_returns_none(self, tmp_path):
        """Provider with non-existent assets dir must not raise."""
        missing_dir = tmp_path / "nonexistent"
        provider = StaticArtworkProvider(assets_dir=missing_dir)
        result = provider.get_artwork("Erithacus rubecula")
        assert result is None

    def test_deleted_file_returns_none(self, tmp_path):
        """If a file is deleted after indexing, get_artwork returns None."""
        path = _make_fake_artwork(tmp_path, "Parus major", ".jpg")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        # Force indexing
        provider.has_artwork("Parus major")
        # Delete the file
        path.unlink()
        result = provider.get_artwork("Parus major")
        assert result is None

    # ------------------------------------------------------------------
    # has_artwork() tests
    # ------------------------------------------------------------------

    def test_has_artwork_true_for_existing(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.has_artwork("Erithacus rubecula") is True

    def test_has_artwork_false_for_missing(self, tmp_path):
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.has_artwork("Unknown birdus") is False

    def test_has_artwork_false_for_empty_name(self, tmp_path):
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.has_artwork("") is False

    # ------------------------------------------------------------------
    # Index / refresh tests
    # ------------------------------------------------------------------

    def test_artwork_count_zero_when_empty(self, tmp_path):
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.artwork_count() == 0

    def test_artwork_count_matches_files(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        _make_fake_artwork(tmp_path, "Parus major", ".jpg")
        _make_fake_artwork(tmp_path, "Turdus merula", ".png")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.artwork_count() == 3

    def test_supported_species_lists_all(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        _make_fake_artwork(tmp_path, "Parus major", ".jpg")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        supported = provider.supported_species
        assert "erithacus rubecula" in supported
        assert "parus major" in supported

    def test_refresh_picks_up_new_files(self, tmp_path):
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.artwork_count() == 0

        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        count = provider.refresh()
        assert count == 1
        assert provider.has_artwork("Erithacus rubecula") is True

    def test_refresh_returns_count(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        _make_fake_artwork(tmp_path, "Parus major", ".png")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        count = provider.refresh()
        assert count == 2

    def test_index_built_lazily(self, tmp_path):
        """Index should not be built until first use."""
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider._indexed is False
        provider.has_artwork("anything")
        assert provider._indexed is True

    # ------------------------------------------------------------------
    # Multiple species tests
    # ------------------------------------------------------------------

    def test_get_artwork_for_species_list(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        _make_fake_artwork(tmp_path, "Parus major", ".jpg")
        provider = StaticArtworkProvider(assets_dir=tmp_path)

        result = provider.get_artwork_for_species_list([
            "Erithacus rubecula",
            "Parus major",
            "Unknown birdus",
        ])

        assert result["Erithacus rubecula"] is not None
        assert result["Parus major"] is not None
        assert result["Unknown birdus"] is None

    def test_missing_artwork_returns_correct_subset(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        provider = StaticArtworkProvider(assets_dir=tmp_path)

        missing = provider.missing_artwork([
            "Erithacus rubecula",
            "Unknown birdus",
            "Parus major",
        ])

        assert "Erithacus rubecula" not in missing
        assert "Unknown birdus" in missing
        assert "Parus major" in missing

    # ------------------------------------------------------------------
    # Extension preference tests
    # ------------------------------------------------------------------

    def test_jpg_found(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpg")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.has_artwork("Erithacus rubecula")

    def test_jpeg_found(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".jpeg")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.has_artwork("Erithacus rubecula")

    def test_png_found(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".png")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.has_artwork("Erithacus rubecula")

    def test_webp_found(self, tmp_path):
        _make_fake_artwork(tmp_path, "Erithacus rubecula", ".webp")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.has_artwork("Erithacus rubecula")

    def test_unsupported_extension_ignored(self, tmp_path):
        # Create a file with an unsupported extension
        (tmp_path / "erithacus_rubecula.bmp").write_bytes(b"fake")
        provider = StaticArtworkProvider(assets_dir=tmp_path)
        assert provider.has_artwork("Erithacus rubecula") is False
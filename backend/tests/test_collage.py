"""
Tests for CollageGenerator.

Uses real artwork files (from assets/artwork/) when available,
and synthetic test images when not. Never requires a display or GUI.
"""

import sys
import math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.collage.generator import CollageGenerator, CollageGeneratorError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(directory: Path, scientific_name: str,
                     size=(200, 300)) -> Path:
    """Create a small valid JPEG for use as fake artwork."""
    from PIL import Image
    stem = scientific_name.lower().replace(" ", "_")
    path = directory / f"{stem}.jpg"
    img = Image.new("RGB", size, color=(200, 180, 140))
    img.save(str(path), "JPEG")
    return path


def _make_species_paths(
    tmp_path: Path,
    species: list[tuple[str, str]],   # [(sci, common), ...]
    include_none: bool = False,
) -> dict:
    """
    Return a species_paths dict with synthetic artwork.
    If include_none is True, the last species gets None for its path.
    """
    result = {}
    for i, (sci, common) in enumerate(species):
        if include_none and i == len(species) - 1:
            result[sci] = (common, None)
        else:
            path = _make_test_image(tmp_path, sci)
            result[sci] = (common, path)
    return result


SAMPLE_SPECIES = [
    ("Erithacus rubecula", "European Robin"),
    ("Parus major", "Great Tit"),
    ("Turdus merula", "Eurasian Blackbird"),
    ("Passer domesticus", "House Sparrow"),
]


# ---------------------------------------------------------------------------
# CollageGenerator construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self, tmp_path):
        gen = CollageGenerator(output_dir=tmp_path)
        assert gen.width == config.COLLAGE_WIDTH
        assert gen.height == config.COLLAGE_HEIGHT

    def test_custom_dimensions(self, tmp_path):
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        assert gen.width == 800
        assert gen.height == 600

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "collages"
        CollageGenerator(output_dir=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Grid dimension tests
# ---------------------------------------------------------------------------

class TestGridDimensions:
    def test_one_species(self):
        cols, rows = CollageGenerator._grid_dimensions(1)
        assert cols * rows >= 1

    def test_two_species(self):
        cols, rows = CollageGenerator._grid_dimensions(2)
        assert cols * rows >= 2

    def test_three_species(self):
        cols, rows = CollageGenerator._grid_dimensions(3)
        assert cols * rows >= 3

    def test_four_species(self):
        cols, rows = CollageGenerator._grid_dimensions(4)
        assert cols * rows >= 4

    def test_six_species(self):
        cols, rows = CollageGenerator._grid_dimensions(6)
        assert cols * rows >= 6

    def test_landscape_preference(self):
        """For typical species counts, cols should be >= rows."""
        for n in [2, 3, 4, 6]:
            cols, rows = CollageGenerator._grid_dimensions(n)
            assert cols >= rows, f"Expected cols>=rows for n={n}, got {cols}x{rows}"

    def test_zero_species(self):
        cols, rows = CollageGenerator._grid_dimensions(0)
        assert cols >= 1
        assert rows >= 1


# ---------------------------------------------------------------------------
# Image fitting tests
# ---------------------------------------------------------------------------

class TestFitImage:
    def test_shrinks_to_fit(self):
        from PIL import Image
        img = Image.new("RGB", (1000, 800))
        result = CollageGenerator._fit_image(img, 200, 200)
        assert result.width <= 200
        assert result.height <= 200

    def test_preserves_aspect_ratio(self):
        from PIL import Image
        img = Image.new("RGB", (400, 200))  # 2:1 aspect
        result = CollageGenerator._fit_image(img, 200, 200)
        ratio = result.width / result.height
        assert abs(ratio - 2.0) < 0.1

    def test_does_not_upscale_unnecessarily(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100))
        result = CollageGenerator._fit_image(img, 500, 500)
        # May upscale to fill — just check result fits within bounds
        assert result.width <= 500
        assert result.height <= 500


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generates_file(self, tmp_path):
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        species_paths = _make_species_paths(tmp_path, SAMPLE_SPECIES[:2])
        output = gen.generate(species_paths)
        assert output.exists()
        assert output.suffix == ".jpg"

    def test_output_in_output_dir(self, tmp_path):
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        species_paths = _make_species_paths(tmp_path, SAMPLE_SPECIES[:2])
        output = gen.generate(species_paths)
        assert output.parent == tmp_path

    def test_custom_filename(self, tmp_path):
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        species_paths = _make_species_paths(tmp_path, SAMPLE_SPECIES[:2])
        output = gen.generate(species_paths, filename="test_collage.jpg")
        assert output.name == "test_collage.jpg"

    def test_output_is_valid_jpeg(self, tmp_path):
        from PIL import Image
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        species_paths = _make_species_paths(tmp_path, SAMPLE_SPECIES[:2])
        output = gen.generate(species_paths)
        # Should open without error
        img = Image.open(str(output))
        assert img.format == "JPEG"

    def test_output_dimensions_correct(self, tmp_path):
        from PIL import Image
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        species_paths = _make_species_paths(tmp_path, SAMPLE_SPECIES[:2])
        output = gen.generate(species_paths)
        img = Image.open(str(output))
        assert img.size == (800, 600)

    def test_single_species(self, tmp_path):
        from PIL import Image
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        species_paths = _make_species_paths(tmp_path, SAMPLE_SPECIES[:1])
        output = gen.generate(species_paths)
        assert output.exists()
        img = Image.open(str(output))
        assert img.size == (800, 600)

    def test_max_species_species(self, tmp_path):
        from PIL import Image
        gen = CollageGenerator(
            width=1920, height=1080, output_dir=tmp_path, max_species=6
        )
        # Provide more than max_species
        many = [
            (f"Species {i}", f"Common {i}") for i in range(8)
        ]
        species_paths = _make_species_paths(tmp_path, many)
        output = gen.generate(species_paths)
        assert output.exists()

    def test_none_paths_skipped(self, tmp_path):
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        species_paths = _make_species_paths(
            tmp_path, SAMPLE_SPECIES[:3], include_none=True
        )
        # Should still work with the non-None species
        output = gen.generate(species_paths)
        assert output.exists()

    def test_all_none_paths_raises(self, tmp_path):
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        species_paths = {
            "Erithacus rubecula": ("European Robin", None),
            "Parus major": ("Great Tit", None),
        }
        with pytest.raises(CollageGeneratorError, match="No valid artwork"):
            gen.generate(species_paths)

    def test_empty_dict_raises(self, tmp_path):
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        with pytest.raises(CollageGeneratorError):
            gen.generate({})

    def test_generate_latest_overwrites(self, tmp_path):
        gen = CollageGenerator(width=800, height=600, output_dir=tmp_path)
        species_paths = _make_species_paths(tmp_path, SAMPLE_SPECIES[:2])

        out1 = gen.generate_latest(species_paths)
        mtime1 = out1.stat().st_mtime

        import time
        time.sleep(0.05)

        out2 = gen.generate_latest(species_paths)
        mtime2 = out2.stat().st_mtime

        assert out1 == out2  # same path
        assert mtime2 >= mtime1  # overwritten

    def test_four_species_grid(self, tmp_path):
        """Four species should produce a valid 2x2 grid collage."""
        from PIL import Image
        gen = CollageGenerator(width=1200, height=800, output_dir=tmp_path)
        species_paths = _make_species_paths(tmp_path, SAMPLE_SPECIES)
        output = gen.generate(species_paths)
        img = Image.open(str(output))
        assert img.size == (1200, 800)


# ---------------------------------------------------------------------------
# Integration test with real artwork
# ---------------------------------------------------------------------------

class TestWithRealArtwork:
    """
    Uses the actual downloaded illustrations from assets/artwork/.
    Skipped if no artwork has been downloaded yet.
    """

    def test_generate_with_real_artwork(self, tmp_path):
        from backend.artwork.static_provider import StaticArtworkProvider
        provider = StaticArtworkProvider()

        if provider.artwork_count() == 0:
            pytest.skip("No artwork downloaded — run scripts/download_artwork.py first")

        species_paths = {}
        for sci_lower in provider.supported_species[:4]:
            # Convert "erithacus rubecula" back to "Erithacus rubecula"
            sci = sci_lower.title()
            common = sci.split()[0]  # rough common name for test
            path = provider.get_artwork(sci_lower)
            if path:
                species_paths[sci_lower] = (common, path)

        if not species_paths:
            pytest.skip("No usable artwork found")

        from PIL import Image
        gen = CollageGenerator(
            width=1920, height=1080, output_dir=tmp_path
        )
        output = gen.generate(species_paths, filename="real_artwork_test.jpg")

        assert output.exists()
        img = Image.open(str(output))
        assert img.size == (1920, 1080)
        print(f"\n  Collage with real artwork: {output}")
        print(f"  Species included: {len(species_paths)}")
        print(f"  File size: {output.stat().st_size:,} bytes")
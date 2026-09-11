"""
Tests for GeneratedArtworkProvider.

Unit tests use mocks — no real ComfyUI or GPU required.
The integration test (marked @pytest.mark.integration) requires
ComfyUI running at http://127.0.0.1:8188.
"""

import sys
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.artwork.generated_provider import (
    GeneratedArtworkProvider,
    PROMPT_TEMPLATE,
    NEGATIVE_PROMPT,
)
from backend.artwork.base import ArtworkProviderError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider(tmp_path):
    return GeneratedArtworkProvider(
        comfyui_url="http://127.0.0.1:8188",
        cache_dir=tmp_path / "generated",
        comfyui_output_dir=tmp_path / "comfyui_output",
        steps=4,
        width=512,
        height=512,
    )


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_construction(self, tmp_path):
        p = GeneratedArtworkProvider(cache_dir=tmp_path)
        assert p.provider_name == "GeneratedArtworkProvider"

    def test_cache_dir_created(self, tmp_path):
        cache = tmp_path / "art_cache"
        GeneratedArtworkProvider(cache_dir=cache)
        assert cache.exists()

    def test_repr(self, provider):
        assert "GeneratedArtworkProvider" in repr(provider)


# ---------------------------------------------------------------------------
# has_artwork tests
# ---------------------------------------------------------------------------

class TestHasArtwork:
    def test_false_when_no_cache(self, provider):
        assert provider.has_artwork("Erithacus rubecula") is False

    def test_true_when_cached(self, provider):
        cached = provider._cache_dir / "erithacus_rubecula.png"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"fake png")
        assert provider.has_artwork("Erithacus rubecula") is True

    def test_empty_name_returns_false(self, provider):
        assert provider.has_artwork("") is False

    def test_supported_species_from_cache(self, provider):
        provider._cache_dir.mkdir(parents=True, exist_ok=True)
        (provider._cache_dir / "parus_major.png").write_bytes(b"x")
        (provider._cache_dir / "turdus_merula.png").write_bytes(b"x")
        species = provider.supported_species
        assert "parus major" in species
        assert "turdus merula" in species


# ---------------------------------------------------------------------------
# get_artwork cache hit tests
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_returns_cached_path(self, provider):
        provider._cache_dir.mkdir(parents=True, exist_ok=True)
        cached = provider._cache_dir / "erithacus_rubecula.png"
        cached.write_bytes(b"fake png")
        result = provider.get_artwork("Erithacus rubecula")
        assert result == cached

    def test_does_not_call_comfyui_on_cache_hit(self, provider):
        provider._cache_dir.mkdir(parents=True, exist_ok=True)
        (provider._cache_dir / "parus_major.png").write_bytes(b"fake")
        with patch.object(provider, "_ping_comfyui") as mock_ping:
            provider.get_artwork("Parus major")
            mock_ping.assert_not_called()


# ---------------------------------------------------------------------------
# Workflow building tests
# ---------------------------------------------------------------------------

class TestWorkflowBuilding:
    def test_prompt_contains_species_names(self, provider):
        wf = provider._build_workflow(
            "Erithacus rubecula", "European Robin", "erithacus_rubecula"
        )
        prompt_text = wf["prompt"]["2"]["inputs"]["text"]
        assert "European Robin" in prompt_text
        assert "Erithacus rubecula" in prompt_text

    def test_negative_prompt_set(self, provider):
        wf = provider._build_workflow("A b", "Common A", "a_b")
        neg = wf["prompt"]["3"]["inputs"]["text"]
        assert len(neg) > 0

    def test_filename_prefix_contains_stem(self, provider):
        wf = provider._build_workflow(
            "Parus major", "Great Tit", "parus_major"
        )
        prefix = wf["prompt"]["7"]["inputs"]["filename_prefix"]
        assert "parus_major" in prefix

    def test_dimensions_respected(self, provider):
        wf = provider._build_workflow("A b", "Common A", "a_b")
        latent = wf["prompt"]["4"]["inputs"]
        assert latent["width"] == 512
        assert latent["height"] == 512

    def test_cfg_is_one_for_flux(self, provider):
        wf = provider._build_workflow("A b", "Common A", "a_b")
        cfg = wf["prompt"]["5"]["inputs"]["cfg"]
        assert cfg == 1.0

    def test_gguf_model_referenced(self, provider):
        wf = provider._build_workflow("A b", "Common A", "a_b")
        unet = wf["prompt"]["1"]["inputs"]["unet_name"]
        assert "gguf" in unet.lower()


# ---------------------------------------------------------------------------
# Ping tests
# ---------------------------------------------------------------------------

class TestPing:
    def test_ping_returns_true_on_success(self, provider):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert provider._ping_comfyui() is True

    def test_ping_returns_false_on_failure(self, provider):
        with patch("urllib.request.urlopen", side_effect=Exception("refused")):
            assert provider._ping_comfyui() is False


# ---------------------------------------------------------------------------
# get_artwork error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_empty_name_raises(self, provider):
        with pytest.raises(ArtworkProviderError):
            provider.get_artwork("")

    def test_comfyui_unreachable_raises(self, provider):
        with patch.object(provider, "_ping_comfyui", return_value=False):
            with pytest.raises(ArtworkProviderError, match="not reachable"):
                provider.get_artwork("Erithacus rubecula", "European Robin")

    def test_returns_none_on_generation_failure(self, provider):
        with patch.object(provider, "_ping_comfyui", return_value=True), \
             patch.object(provider, "_generate_via_comfyui", return_value=None):
            result = provider.get_artwork("Erithacus rubecula", "European Robin")
            assert result is None


# ---------------------------------------------------------------------------
# clear_cache tests
# ---------------------------------------------------------------------------

class TestClearCache:
    def test_clear_specific_species(self, provider):
        provider._cache_dir.mkdir(parents=True, exist_ok=True)
        f1 = provider._cache_dir / "erithacus_rubecula.png"
        f2 = provider._cache_dir / "parus_major.png"
        f1.write_bytes(b"x")
        f2.write_bytes(b"x")

        deleted = provider.clear_cache("Erithacus rubecula")
        assert deleted == 1
        assert not f1.exists()
        assert f2.exists()

    def test_clear_all(self, provider):
        provider._cache_dir.mkdir(parents=True, exist_ok=True)
        for name in ["robin.png", "tit.png", "blackbird.png"]:
            (provider._cache_dir / name).write_bytes(b"x")

        deleted = provider.clear_cache()
        assert deleted == 3
        assert list(provider._cache_dir.glob("*.png")) == []


# ---------------------------------------------------------------------------
# Integration test — requires ComfyUI running
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGeneratedProviderIntegration:
    """
    Requires ComfyUI running at http://127.0.0.1:8188 with:
      - flux1-dev-Q4_K_S.gguf in models/unet/
      - clip_l.safetensors + t5xxl_fp8_e4m3fn.safetensors in models/clip/
      - ae.safetensors in models/vae/
      - ComfyUI-GGUF custom node installed

    Run with:
        pytest -m integration backend/tests/test_generated_provider.py -v -s
    """

    @pytest.fixture(scope="class")
    def provider(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("generated_art")
        return GeneratedArtworkProvider(
            cache_dir=tmp,
            steps=4,    # fast draft for testing
            width=512,
            height=512,
        )

    def test_comfyui_reachable(self, provider):
        assert provider._ping_comfyui(), (
            "ComfyUI must be running at http://127.0.0.1:8188"
        )

    def test_generates_robin(self, provider):
        path = provider.get_artwork(
            "Erithacus rubecula", "European Robin"
        )
        assert path is not None, "Generation failed"
        assert path.exists(), f"File not found: {path}"
        assert path.suffix == ".png"
        assert path.stat().st_size > 1000
        print(f"\n  Generated: {path} ({path.stat().st_size:,} bytes)")

    def test_cache_hit_on_second_call(self, provider):
        start = time.time()
        path = provider.get_artwork("Erithacus rubecula", "European Robin")
        elapsed = time.time() - start
        assert path is not None
        assert elapsed < 1.0, "Cache hit should be instant"
        print(f"\n  Cache hit in {elapsed:.3f}s")
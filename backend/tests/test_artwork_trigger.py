"""
Tests for ArtworkTrigger.
"""

import sys
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.detection.artwork_trigger import ArtworkTrigger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_provider(tmp_path):
    provider = MagicMock()
    provider.has_artwork.return_value = False
    provider._ping_comfyui.return_value = True

    # Simulate generating a real file
    def fake_get_artwork(sci, common_name=None):
        path = tmp_path / f"{sci.replace(' ', '_')}.png"
        path.write_bytes(b"fake png")
        return path

    provider.get_artwork.side_effect = fake_get_artwork
    return provider


@pytest.fixture
def trigger():
    return ArtworkTrigger(enabled=True)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_not_running_before_start(self, trigger):
        assert trigger.is_running is False

    def test_disabled_trigger(self):
        t = ArtworkTrigger(enabled=False)
        t.start()
        assert t.is_running is False

    def test_empty_sets_on_init(self, trigger):
        assert trigger.species_completed == set()
        assert trigger.species_in_progress == set()


# ---------------------------------------------------------------------------
# Start / stop tests
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_start_with_static_backend(self):
        with patch("backend.detection.artwork_trigger.config") as mock_config:
            mock_config.ARTWORK_BACKEND = "static"
            t = ArtworkTrigger(enabled=True)
            t.start()
            assert t.is_running is False  # static backend disables trigger

    def test_start_with_generated_backend_comfyui_unreachable(self):
        with patch("backend.detection.artwork_trigger.config") as mock_config, \
             patch(
                 "backend.detection.artwork_trigger.GeneratedArtworkProvider"
             ) as MockProvider:
            mock_config.ARTWORK_BACKEND = "generated"
            mock_config.COMFYUI_URL = "http://127.0.0.1:8188"
            mock_config.COMFYUI_OUTPUT_DIR = "C:\\fake"

            instance = MockProvider.return_value
            instance._ping_comfyui.return_value = False

            t = ArtworkTrigger(enabled=True)
            t.start()
            assert t.is_running is False  # ComfyUI unreachable

    def test_start_with_generated_backend_comfyui_reachable(self):
        with patch("backend.detection.artwork_trigger.config") as mock_config, \
             patch(
                 "backend.detection.artwork_trigger.GeneratedArtworkProvider"
             ) as MockProvider:
            mock_config.ARTWORK_BACKEND = "generated"
            mock_config.COMFYUI_URL = "http://127.0.0.1:8188"
            mock_config.COMFYUI_OUTPUT_DIR = "C:\\fake"

            instance = MockProvider.return_value
            instance._ping_comfyui.return_value = True

            t = ArtworkTrigger(enabled=True)
            t.start()
            assert t.is_running is True
            t.stop()

    def test_stop_sets_not_running(self):
        with patch("backend.detection.artwork_trigger.config") as mock_config, \
             patch(
                 "backend.detection.artwork_trigger.GeneratedArtworkProvider"
             ) as MockProvider:
            mock_config.ARTWORK_BACKEND = "generated"
            mock_config.COMFYUI_URL = "http://127.0.0.1:8188"
            mock_config.COMFYUI_OUTPUT_DIR = "C:\\fake"

            instance = MockProvider.return_value
            instance._ping_comfyui.return_value = True

            t = ArtworkTrigger(enabled=True)
            t.start()
            t.stop()
            assert t.is_running is False


# ---------------------------------------------------------------------------
# on_detection tests
# ---------------------------------------------------------------------------

class TestOnDetection:
    def _running_trigger(self, mock_provider):
        """Return a trigger that is already running with a mock provider."""
        t = ArtworkTrigger(enabled=True)
        t._provider = mock_provider
        t._running = True
        return t

    def test_on_detection_when_not_running_is_noop(self, mock_provider):
        t = ArtworkTrigger(enabled=True)
        t._running = False
        t.on_detection("Erithacus rubecula", "European Robin")
        mock_provider.get_artwork.assert_not_called()

    def test_on_detection_skips_cached_species(self, mock_provider):
        mock_provider.has_artwork.return_value = True
        t = self._running_trigger(mock_provider)
        t.on_detection("Erithacus rubecula", "European Robin")
        time.sleep(0.1)
        mock_provider.get_artwork.assert_not_called()
        assert "Erithacus rubecula" in t.species_completed

    def test_on_detection_skips_in_progress(self, mock_provider):
        t = self._running_trigger(mock_provider)
        t._in_progress.add("Erithacus rubecula")
        t.on_detection("Erithacus rubecula", "European Robin")
        time.sleep(0.1)
        mock_provider.get_artwork.assert_not_called()

    def test_on_detection_skips_already_completed(self, mock_provider):
        t = self._running_trigger(mock_provider)
        t._completed.add("Erithacus rubecula")
        t.on_detection("Erithacus rubecula", "European Robin")
        time.sleep(0.1)
        mock_provider.get_artwork.assert_not_called()

    def test_on_detection_triggers_generation(self, mock_provider):
        t = self._running_trigger(mock_provider)
        t.on_detection("Erithacus rubecula", "European Robin")
        # Wait for background thread
        deadline = time.time() + 5.0
        while "Erithacus rubecula" not in t.species_completed:
            if time.time() > deadline:
                break
            time.sleep(0.05)
        assert "Erithacus rubecula" in t.species_completed
        mock_provider.get_artwork.assert_called_once_with(
            "Erithacus rubecula", common_name="European Robin"
        )

    def test_same_species_not_generated_twice(self, mock_provider):
        t = self._running_trigger(mock_provider)
        t.on_detection("Erithacus rubecula", "European Robin")
        t.on_detection("Erithacus rubecula", "European Robin")
        time.sleep(0.5)
        # get_artwork should be called at most once
        assert mock_provider.get_artwork.call_count <= 1

    def test_different_species_both_generated(self, mock_provider):
        t = self._running_trigger(mock_provider)
        t.on_detection("Erithacus rubecula", "European Robin")
        t.on_detection("Parus major", "Great Tit")

        deadline = time.time() + 10.0
        while len(t.species_completed) < 2:
            if time.time() > deadline:
                break
            time.sleep(0.05)

        assert "Erithacus rubecula" in t.species_completed
        assert "Parus major" in t.species_completed

    def test_generation_failure_removes_from_in_progress(self, mock_provider):
        mock_provider.get_artwork.side_effect = Exception("ComfyUI error")
        t = self._running_trigger(mock_provider)
        t.on_detection("Erithacus rubecula", "European Robin")
        time.sleep(0.5)
        # Must not stay stuck in in_progress
        assert "Erithacus rubecula" not in t.species_in_progress
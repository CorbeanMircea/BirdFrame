"""
Tests for config.py — verifies that all required settings exist,
have the right types, and that the directories are created.
"""

import sys
from pathlib import Path

# Make sure the project root is on the path so `import config` works
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config


class TestPaths:
    def test_base_dir_exists(self):
        assert config.BASE_DIR.exists(), "BASE_DIR must exist"

    def test_data_dir_created(self):
        assert config.DATA_DIR.exists(), "DATA_DIR must be created on import"

    def test_assets_dir_created(self):
        assert config.ASSETS_DIR.exists(), "ASSETS_DIR must be created on import"

    def test_collage_dir_created(self):
        assert config.COLLAGE_DIR.exists(), "COLLAGE_DIR must be created on import"

    def test_audio_dir_created(self):
        assert config.AUDIO_DIR.exists(), "AUDIO_DIR must be created on import"


class TestDatabaseConfig:
    def test_database_url_is_string(self):
        assert isinstance(config.DATABASE_URL, str)

    def test_database_url_contains_sqlite(self):
        assert "sqlite" in config.DATABASE_URL

    def test_database_url_points_inside_data_dir(self):
        # Strip the sqlite:/// prefix and check the path
        db_path = config.DATABASE_URL.replace("sqlite:///", "")
        assert str(config.DATA_DIR) in db_path


class TestAudioConfig:
    def test_sample_rate_is_positive_int(self):
        assert isinstance(config.AUDIO_SAMPLE_RATE, int)
        assert config.AUDIO_SAMPLE_RATE > 0

    def test_channels_is_positive_int(self):
        assert isinstance(config.AUDIO_CHANNELS, int)
        assert config.AUDIO_CHANNELS >= 1

    def test_chunk_duration_is_positive_float(self):
        assert isinstance(config.AUDIO_CHUNK_DURATION, float)
        assert config.AUDIO_CHUNK_DURATION > 0

    def test_chunk_overlap_less_than_duration(self):
        assert config.AUDIO_CHUNK_OVERLAP < config.AUDIO_CHUNK_DURATION


class TestDetectorConfig:
    def test_energy_threshold_non_negative(self):
        assert config.DETECTOR_ENERGY_THRESHOLD >= 0

    def test_freq_band_valid(self):
        assert config.DETECTOR_FREQ_MIN < config.DETECTOR_FREQ_MAX
        assert config.DETECTOR_FREQ_MIN > 0


class TestIdentifierConfig:
    def test_backend_is_string(self):
        assert isinstance(config.IDENTIFIER_BACKEND, str)

    def test_default_backend_is_mock(self):
        # In a clean environment without the env var set, default is "mock"
        assert config.IDENTIFIER_BACKEND in ("mock", "birdnet")

    def test_min_confidence_in_range(self):
        assert 0.0 <= config.IDENTIFIER_MIN_CONFIDENCE <= 1.0


class TestCollageConfig:
    def test_max_species_positive(self):
        assert config.COLLAGE_MAX_SPECIES > 0

    def test_dimensions_positive(self):
        assert config.COLLAGE_WIDTH > 0
        assert config.COLLAGE_HEIGHT > 0

    def test_refresh_minutes_positive(self):
        assert config.COLLAGE_REFRESH_MINUTES > 0


class TestAPIConfig:
    def test_host_is_string(self):
        assert isinstance(config.API_HOST, str)

    def test_port_in_valid_range(self):
        assert 1024 <= config.API_PORT <= 65535
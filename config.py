"""
BirdFrame configuration.

All tunable parameters live here. Import this module anywhere in the
project rather than hard-coding values.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the project (the directory that contains this file)
BASE_DIR = Path(__file__).parent.resolve()

# Persistent data
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Bird artwork illustrations
ASSETS_DIR = BASE_DIR / "assets" / "artwork"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Generated collage images
COLLAGE_DIR = BASE_DIR / "data" / "collages"
COLLAGE_DIR.mkdir(parents=True, exist_ok=True)

# Optional retained audio clips
AUDIO_DIR = BASE_DIR / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASE_URL = f"sqlite:///{DATA_DIR / 'birdframe.db'}"

# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

# Input device index (None = system default)
AUDIO_DEVICE_INDEX: int | None = None

# Samples per second — BirdNET expects 48 000; simpler detectors can use 16 000
AUDIO_SAMPLE_RATE: int = 48_000

# Mono input
AUDIO_CHANNELS: int = 1

# Duration of each analysis chunk in seconds
AUDIO_CHUNK_DURATION: float = 3.0

# Overlap between consecutive chunks in seconds (reduces missed calls at edges)
AUDIO_CHUNK_OVERLAP: float = 0.5

# ---------------------------------------------------------------------------
# Bird detector (energy-based pre-filter)
# ---------------------------------------------------------------------------

# RMS energy threshold below which a chunk is considered silence
DETECTOR_ENERGY_THRESHOLD: float = 0.001

# Frequency band of interest for bird song (Hz)
DETECTOR_FREQ_MIN: int = 1_000
DETECTOR_FREQ_MAX: int = 10_000

# ---------------------------------------------------------------------------
# Bird identification
# ---------------------------------------------------------------------------

# Which identifier implementation to use:
#   "mock"    — MockBirdIdentifier (development / testing)
#   "birdnet" — BirdNetIdentifier  (Phase 5)
IDENTIFIER_BACKEND: str = os.getenv("BIRDFRAME_IDENTIFIER", "mock")

# Minimum confidence (0–1) required to persist a detection
IDENTIFIER_MIN_CONFIDENCE: float = 0.5

# ---------------------------------------------------------------------------
# Detection grouping
# ---------------------------------------------------------------------------

# Detections of the same species within this many seconds are merged
GROUPING_GAP_SECONDS: float = 60.0

# ---------------------------------------------------------------------------
# Artwork
# ---------------------------------------------------------------------------

# Which artwork provider to use:
#   "static"    — StaticArtworkProvider  (looks up files in ASSETS_DIR)
#   "generated" — GeneratedArtworkProvider (Phase 11)
ARTWORK_BACKEND: str = "static"

# ---------------------------------------------------------------------------
# Collage
# ---------------------------------------------------------------------------

# Maximum number of species shown in one collage
COLLAGE_MAX_SPECIES: int = 6

# Output image dimensions in pixels
COLLAGE_WIDTH: int = 1920
COLLAGE_HEIGHT: int = 1080

# How many minutes between automatic collage regeneration
COLLAGE_REFRESH_MINUTES: int = 5

# ---------------------------------------------------------------------------
# "Heard recently" window
# ---------------------------------------------------------------------------

# Species detected within this many hours count as "recently heard"
HEARD_RECENTLY_HOURS: int = 24

# ---------------------------------------------------------------------------
# Audio retention
# ---------------------------------------------------------------------------

# Set to True to save the raw audio clip for every detection
AUDIO_RETAIN_CLIPS: bool = False

# Delete retained clips older than this many days (0 = keep forever)
AUDIO_RETENTION_DAYS: int = 7

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------

API_HOST: str = os.getenv("BIRDFRAME_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("BIRDFRAME_PORT", "8000"))
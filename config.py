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

BASE_DIR = Path(__file__).parent.resolve()

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ASSETS_DIR = BASE_DIR / "assets" / "artwork"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

COLLAGE_DIR = BASE_DIR / "data" / "collages"
COLLAGE_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIR = BASE_DIR / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASE_URL = f"sqlite:///{DATA_DIR / 'birdframe.db'}"

# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

AUDIO_DEVICE_INDEX: int | None = None
AUDIO_SAMPLE_RATE: int = 48_000
AUDIO_CHANNELS: int = 1
AUDIO_CHUNK_DURATION: float = 3.0
AUDIO_CHUNK_OVERLAP: float = 0.5

# ---------------------------------------------------------------------------
# Bird detector
# ---------------------------------------------------------------------------

DETECTOR_ENERGY_THRESHOLD: float = 0.001
DETECTOR_FREQ_MIN: int = 1_000
DETECTOR_FREQ_MAX: int = 10_000

# ---------------------------------------------------------------------------
# Bird identification
# ---------------------------------------------------------------------------

IDENTIFIER_BACKEND: str = os.getenv("BIRDFRAME_IDENTIFIER", "mock")
IDENTIFIER_MIN_CONFIDENCE: float = 0.5

# ---------------------------------------------------------------------------
# Detection grouping
# ---------------------------------------------------------------------------

GROUPING_GAP_SECONDS: float = 60.0

# ---------------------------------------------------------------------------
# Artwork
# ---------------------------------------------------------------------------

# Which artwork provider to use:
#   "static"    — StaticArtworkProvider  (pre-downloaded illustrations)
#   "generated" — GeneratedArtworkProvider (ComfyUI + Flux.1)
ARTWORK_BACKEND: str = os.getenv("BIRDFRAME_ARTWORK", "static")

# ComfyUI API URL (used by GeneratedArtworkProvider)
COMFYUI_URL: str = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# ComfyUI output directory (where ComfyUI saves generated images)
COMFYUI_OUTPUT_DIR: str = os.getenv(
    "COMFYUI_OUTPUT_DIR",
    r"C:\Users\Gaming_PC\ComfyUI_windows_portable\ComfyUI\output",
)

# ---------------------------------------------------------------------------
# Collage
# ---------------------------------------------------------------------------

COLLAGE_MAX_SPECIES: int = 6
COLLAGE_WIDTH: int = 1920
COLLAGE_HEIGHT: int = 1080
COLLAGE_REFRESH_MINUTES: int = 5

# ---------------------------------------------------------------------------
# "Heard recently" window
# ---------------------------------------------------------------------------

HEARD_RECENTLY_HOURS: int = 24

# ---------------------------------------------------------------------------
# Audio retention
# ---------------------------------------------------------------------------

AUDIO_RETAIN_CLIPS: bool = False
AUDIO_RETENTION_DAYS: int = 7

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------

API_HOST: str = os.getenv("BIRDFRAME_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("BIRDFRAME_PORT", "8000"))
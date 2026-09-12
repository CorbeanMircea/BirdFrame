"""
ArtworkProvider factory.

Returns the configured artwork provider based on config.ARTWORK_BACKEND.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.artwork.base import ArtworkProvider

logger = logging.getLogger(__name__)


def get_artwork_provider(backend: str | None = None) -> ArtworkProvider:
    """
    Return the configured ArtworkProvider instance.

    With ARTWORK_BACKEND=generated, only GeneratedArtworkProvider is used.
    Static illustrations are NOT used as fallback — generated artwork only.
    """
    backend = backend or config.ARTWORK_BACKEND

    if backend == "generated":
        from backend.artwork.generated_provider import GeneratedArtworkProvider
        provider = GeneratedArtworkProvider(
            comfyui_url=config.COMFYUI_URL,
            comfyui_output_dir=Path(config.COMFYUI_OUTPUT_DIR),
            steps=28,
            width=768,
            height=768,
        )
        logger.info("Using GeneratedArtworkProvider (ComfyUI + Flux.1, 28 steps)")
        return provider

    from backend.artwork.static_provider import StaticArtworkProvider
    provider = StaticArtworkProvider()
    logger.info("Using StaticArtworkProvider")
    return provider
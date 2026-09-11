"""
ArtworkProvider factory.

Returns the configured artwork provider based on config.ARTWORK_BACKEND.

Usage:
    from backend.artwork.provider_factory import get_artwork_provider
    provider = get_artwork_provider()
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

    Parameters
    ----------
    backend : str | None
        Override config.ARTWORK_BACKEND. One of "static" or "generated".

    Returns
    -------
    ArtworkProvider — StaticArtworkProvider or GeneratedArtworkProvider.
    """
    backend = backend or config.ARTWORK_BACKEND

    if backend == "generated":
        from backend.artwork.generated_provider import GeneratedArtworkProvider
        provider = GeneratedArtworkProvider(
            comfyui_url=config.COMFYUI_URL,
            comfyui_output_dir=Path(config.COMFYUI_OUTPUT_DIR),
            steps=20,
            width=768,
            height=768,
        )
        logger.info("Using GeneratedArtworkProvider (ComfyUI + Flux.1)")
        return provider

    # Default: static
    from backend.artwork.static_provider import StaticArtworkProvider
    provider = StaticArtworkProvider()
    logger.info("Using StaticArtworkProvider")
    return provider
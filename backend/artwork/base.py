"""
ArtworkProvider — abstract base class for all artwork backends.

Every artwork backend must subclass ArtworkProvider and implement
get_artwork() and has_artwork().

The rest of the application only imports from this module, never from
a concrete backend directly. Switching artwork backends is a one-line
config change.

Supported backends (current and planned):
  StaticArtworkProvider    — looks up pre-downloaded illustrations
                             in ASSETS_DIR (Phase 8)
  GeneratedArtworkProvider — calls an image generation API
                             (Phase 11, optional)

Usage:

    from backend.artwork.base import ArtworkProvider
    from backend.artwork.static_provider import StaticArtworkProvider

    provider = StaticArtworkProvider()
    path = provider.get_artwork("Erithacus rubecula")
    if path:
        print(f"Artwork at: {path}")
"""

import sys
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ArtworkProviderError(Exception):
    """Raised when an ArtworkProvider cannot fulfil a request."""
    pass


class ArtworkProvider(ABC):
    """
    Abstract base class for bird artwork providers.

    Subclasses must implement:
        get_artwork(scientific_name) -> Path | None
        has_artwork(scientific_name) -> bool

    Subclasses may optionally override:
        provider_name  -> str  (property)
        supported_species -> list[str]  (property)
    """

    # ------------------------------------------------------------------
    # Properties subclasses may override
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Human-readable name for this provider."""
        return self.__class__.__name__

    @property
    def supported_species(self) -> list[str]:
        """
        List of scientific names this provider has artwork for.
        Empty list means "unknown" (provider will be queried directly).
        Override for providers that maintain an index.
        """
        return []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_artwork(self, scientific_name: str) -> Optional[Path]:
        """
        Return the Path to artwork for *scientific_name*, or None.

        The returned path must point to an existing image file.
        Supported formats: JPEG, PNG, WebP.

        Parameters
        ----------
        scientific_name : str
            Latin binomial, e.g. "Erithacus rubecula".

        Returns
        -------
        Path | None
            Absolute path to the image file, or None if no artwork
            is available for this species.

        Raises
        ------
        ArtworkProviderError
            If the provider encounters an unexpected error (not the same
            as simply not having artwork for a species, which returns None).
        """
        ...

    @abstractmethod
    def has_artwork(self, scientific_name: str) -> bool:
        """
        Return True if artwork is available for *scientific_name*.

        This is a cheaper check than get_artwork() — it should not
        download or generate anything, just check availability.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_artwork_for_species_list(
        self,
        scientific_names: list[str],
    ) -> dict[str, Optional[Path]]:
        """
        Return a dict mapping scientific_name → Path | None for each
        name in *scientific_names*.

        Calls get_artwork() for each name. Subclasses may override this
        for batch efficiency (e.g., a single DB query).
        """
        result: dict[str, Optional[Path]] = {}
        for name in scientific_names:
            try:
                result[name] = self.get_artwork(name)
            except ArtworkProviderError as exc:
                logger.warning(
                    "ArtworkProvider error for %r: %s", name, exc
                )
                result[name] = None
        return result

    def missing_artwork(self, scientific_names: list[str]) -> list[str]:
        """
        Return the subset of *scientific_names* that have no artwork.

        Useful for batch-sourcing missing illustrations.
        """
        return [n for n in scientific_names if not self.has_artwork(n)]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name!r}>"
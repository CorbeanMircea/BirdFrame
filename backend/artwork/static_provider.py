"""
StaticArtworkProvider — serves pre-downloaded bird illustrations from
the local assets directory.

Artwork files must be placed in config.ASSETS_DIR with filenames
derived from the species scientific name:

    Erithacus rubecula  →  erithacus_rubecula.jpg  (or .png / .webp)

The provider scans ASSETS_DIR on first use and caches the index in
memory. Call refresh() to pick up newly added files without restarting.

Supported file extensions: .jpg, .jpeg, .png, .webp

Desired aesthetic (for artwork sourcing):
    - Vintage natural history illustration
    - Scientifically recognisable, detailed anatomy
    - Museum / natural history collection feeling
    - Subtle paper texture
    - No watermark, no unnecessary text

Good public-domain sources:
    - Wikimedia Commons (John Gould plates, Naumann plates)
    - Biodiversity Heritage Library (archive.org)
    - Project Gutenberg illustrated bird books

Usage:

    from backend.artwork.static_provider import StaticArtworkProvider

    provider = StaticArtworkProvider()
    path = provider.get_artwork("Erithacus rubecula")
    if path:
        print(f"Found: {path}")
    else:
        print("No artwork available")
"""

import sys
import logging
import re
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.artwork.base import ArtworkProvider, ArtworkProviderError

logger = logging.getLogger(__name__)

# Supported image formats, in preference order
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def scientific_name_to_filename(scientific_name: str) -> str:
    """
    Convert a scientific name to a safe filename stem.

    "Erithacus rubecula"  →  "erithacus_rubecula"
    "Parus major"         →  "parus_major"

    Rules:
    - Lowercase
    - Spaces replaced with underscores
    - All non-alphanumeric characters (except underscore) removed
    """
    stem = scientific_name.lower().strip()
    stem = stem.replace(" ", "_")
    stem = re.sub(r"[^a-z0-9_]", "", stem)
    return stem


class StaticArtworkProvider(ArtworkProvider):
    """
    Serves pre-downloaded bird illustrations from ASSETS_DIR.

    Parameters
    ----------
    assets_dir : Path | None
        Directory to look for artwork files.
        Defaults to config.ASSETS_DIR.
    """

    def __init__(
        self,
        assets_dir: Optional[Path] = None,
    ) -> None:
        self._assets_dir = assets_dir or config.ASSETS_DIR
        self._index: dict[str, Path] = {}   # scientific_name → Path
        self._indexed = False

        logger.debug(
            "StaticArtworkProvider initialised: assets_dir=%s",
            self._assets_dir,
        )

    # ------------------------------------------------------------------
    # ArtworkProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "StaticArtworkProvider"

    @property
    def supported_species(self) -> list[str]:
        """Return all scientific names with artwork available."""
        self._ensure_indexed()
        return list(self._index.keys())

    def get_artwork(self, scientific_name: str) -> Optional[Path]:
        """
        Return the Path to the illustration for *scientific_name*, or None.

        The file must exist in ASSETS_DIR with a name matching the
        scientific name (case-insensitive, spaces → underscores).
        """
        if not scientific_name or not scientific_name.strip():
            raise ArtworkProviderError(
                "scientific_name must not be empty."
            )

        self._ensure_indexed()
        normalised = scientific_name.strip().lower()
        path = self._index.get(normalised)

        if path is not None and path.exists():
            logger.debug("Artwork found for %r: %s", scientific_name, path)
            return path

        if path is not None and not path.exists():
            # File was indexed but deleted — remove from index
            logger.warning(
                "Artwork file was deleted: %s — removing from index.", path
            )
            del self._index[normalised]

        logger.debug("No artwork found for %r", scientific_name)
        return None

    def has_artwork(self, scientific_name: str) -> bool:
        """Return True if an illustration exists for *scientific_name*."""
        if not scientific_name or not scientific_name.strip():
            return False
        self._ensure_indexed()
        normalised = scientific_name.strip().lower()
        path = self._index.get(normalised)
        return path is not None and path.exists()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def refresh(self) -> int:
        """
        Re-scan ASSETS_DIR and rebuild the index.

        Returns the number of artwork files found.
        Call this after adding new files to pick them up without
        restarting the application.
        """
        self._index = {}
        self._indexed = False
        self._build_index()
        return len(self._index)

    def artwork_count(self) -> int:
        """Return the number of species with artwork available."""
        self._ensure_indexed()
        return len(self._index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_indexed(self) -> None:
        """Build the index on first use."""
        if not self._indexed:
            self._build_index()

    def _build_index(self) -> None:
        """
        Scan ASSETS_DIR for supported image files and build the index.

        For each file, the scientific name is inferred by reversing the
        filename normalisation:
            "erithacus_rubecula.jpg" → "erithacus rubecula"
                                     → stored as "erithacus rubecula"

        Lookup is always done with the normalised (lowercase) name.
        """
        if not self._assets_dir.exists():
            logger.warning(
                "Assets directory does not exist: %s", self._assets_dir
            )
            self._indexed = True
            return

        found = 0
        for ext in SUPPORTED_EXTENSIONS:
            for path in self._assets_dir.glob(f"*{ext}"):
                # Infer the normalised scientific name from the filename
                stem = path.stem.lower()
                # Convert underscores back to spaces for lookup normalisation
                # but we store by the lowercase form that scientific_name
                # lookups will also produce
                self._index[stem.replace("_", " ")] = path
                found += 1

        self._indexed = True
        logger.info(
            "StaticArtworkProvider indexed %d file(s) from %s",
            found, self._assets_dir,
        )

    def _path_for(self, scientific_name: str) -> Optional[Path]:
        """
        Return the expected Path for *scientific_name* regardless of
        whether the file exists. Used for testing and diagnostics.
        """
        stem = scientific_name_to_filename(scientific_name)
        for ext in SUPPORTED_EXTENSIONS:
            candidate = self._assets_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None
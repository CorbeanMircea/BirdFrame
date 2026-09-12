"""
ArtworkTrigger — automatically generates artwork for newly detected species.

When DetectionService saves a detection for a species that has no artwork,
ArtworkTrigger spawns a background thread to generate it via ComfyUI.

Design principles:
  - Never blocks the audio pipeline (runs in daemon threads)
  - Never generates the same species twice simultaneously (dedup lock)
  - Silently skips if ComfyUI is unavailable
  - Only generates for GeneratedArtworkProvider (static has fixed artwork)
"""

import sys
import logging
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config

logger = logging.getLogger(__name__)

# Import at module level so tests can patch it
try:
    from backend.artwork.generated_provider import GeneratedArtworkProvider
    _GENERATED_AVAILABLE = True
except ImportError:
    GeneratedArtworkProvider = None  # type: ignore
    _GENERATED_AVAILABLE = False


class ArtworkTrigger:
    """
    Listens for new species detections and triggers artwork generation.

    Parameters
    ----------
    max_concurrent : int
        Maximum simultaneous generation threads (1 = one GPU).
    enabled : bool
        If False, on_detection() is a no-op.
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._semaphore = threading.Semaphore(max_concurrent)
        self._in_progress: set[str] = set()
        self._completed: set[str] = set()
        self._lock = threading.Lock()
        self._provider = None
        self._running = False

        logger.debug(
            "ArtworkTrigger created: enabled=%s max_concurrent=%d",
            enabled, max_concurrent,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Initialise the artwork provider.
        Only activates if ARTWORK_BACKEND=generated and ComfyUI is reachable.
        """
        if not self._enabled:
            return

        backend = config.ARTWORK_BACKEND
        if backend != "generated":
            logger.info(
                "ArtworkTrigger: backend=%s — auto-generation disabled "
                "(only active with ARTWORK_BACKEND=generated).",
                backend,
            )
            self._running = False
            return

        if not _GENERATED_AVAILABLE or GeneratedArtworkProvider is None:
            logger.warning(
                "ArtworkTrigger: GeneratedArtworkProvider not available."
            )
            self._running = False
            return

        try:
            self._provider = GeneratedArtworkProvider(
                comfyui_url=config.COMFYUI_URL,
                comfyui_output_dir=Path(config.COMFYUI_OUTPUT_DIR),
                steps=20,
                width=768,
                height=768,
            )

            if not self._provider._ping_comfyui():
                logger.warning(
                    "ArtworkTrigger: ComfyUI not reachable at %s. "
                    "Auto-generation disabled.",
                    config.COMFYUI_URL,
                )
                self._provider = None
                self._running = False
                return

            self._running = True
            logger.info(
                "ArtworkTrigger started: ComfyUI reachable at %s",
                config.COMFYUI_URL,
            )

        except Exception as exc:
            logger.warning("ArtworkTrigger failed to start: %s", exc)
            self._running = False

    def stop(self) -> None:
        """Stop accepting new generation requests."""
        self._running = False
        logger.info(
            "ArtworkTrigger stopped. Generated this session: %d species.",
            len(self._completed),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_detection(
        self,
        scientific_name: str,
        common_name: str,
    ) -> None:
        """
        Called by DetectionService when a species is detected.
        Spawns a background thread if artwork is needed. Never blocks.
        """
        if not self._running or self._provider is None:
            return

        with self._lock:
            if scientific_name in self._completed:
                return
            if scientific_name in self._in_progress:
                return
            if self._provider.has_artwork(scientific_name):
                self._completed.add(scientific_name)
                return
            self._in_progress.add(scientific_name)

        logger.info(
            "New species — queuing artwork generation: %s (%s)",
            scientific_name, common_name,
        )

        thread = threading.Thread(
            target=self._generate,
            args=(scientific_name, common_name),
            daemon=True,
            name=f"artwork-{scientific_name.replace(' ', '_')}",
        )
        thread.start()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def species_completed(self) -> set[str]:
        return set(self._completed)

    @property
    def species_in_progress(self) -> set[str]:
        return set(self._in_progress)

    # ------------------------------------------------------------------
    # Background generation
    # ------------------------------------------------------------------

    def _generate(self, scientific_name: str, common_name: str) -> None:
        """Generate artwork in a background thread."""
        with self._semaphore:
            logger.info(
                "Generating artwork: %s (%s)…", scientific_name, common_name
            )
            try:
                path = self._provider.get_artwork(
                    scientific_name,
                    common_name=common_name,
                )
                if path and path.exists():
                    logger.info(
                        "Artwork ready: %s → %s (%d KB)",
                        scientific_name, path.name,
                        path.stat().st_size // 1024,
                    )
                    with self._lock:
                        self._completed.add(scientific_name)
                else:
                    logger.warning(
                        "No file returned for %s", scientific_name
                    )
            except Exception as exc:
                logger.error(
                    "Artwork generation failed for %s: %s",
                    scientific_name, exc,
                )
            finally:
                with self._lock:
                    self._in_progress.discard(scientific_name)
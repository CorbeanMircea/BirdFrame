"""
GeneratedArtworkProvider — generates bird illustrations via ComfyUI
and removes the background to produce transparent cut-out PNGs.

Pipeline per species:
    1. Check local cache (assets/artwork/generated/<stem>.png)
    2. If not cached: call ComfyUI HTTP API with a Flux.1 workflow
    3. Poll ComfyUI until generation completes
    4. Download the generated image
    5. Remove background with rembg → transparent PNG
    6. Save to cache and return the path

The provider is a drop-in replacement for StaticArtworkProvider.
Switch in config.py:  ARTWORK_BACKEND = "generated"

ComfyUI must be running at COMFYUI_URL before calling get_artwork().

Usage:

    from backend.artwork.generated_provider import GeneratedArtworkProvider

    provider = GeneratedArtworkProvider()
    path = provider.get_artwork("Erithacus rubecula")
    # path → assets/artwork/generated/erithacus_rubecula.png
"""

import sys
import json
import time
import uuid
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.artwork.base import ArtworkProvider, ArtworkProviderError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ComfyUI API base URL
COMFYUI_URL = "http://127.0.0.1:8188"

# Where generated (background-removed) PNGs are cached
GENERATED_DIR = config.ASSETS_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# ComfyUI output directory — where ComfyUI saves generated images
COMFYUI_OUTPUT_DIR = Path(
    r"C:\Users\Gaming_PC\ComfyUI_windows_portable\ComfyUI\output"
)

# Generation settings
GENERATION_TIMEOUT = 120   # seconds to wait for ComfyUI
POLL_INTERVAL     = 2.0    # seconds between status polls

# Prompt template — {common_name} and {scientific_name} are substituted
PROMPT_TEMPLATE = (
    "vintage natural history illustration of a single {common_name} "
    "({scientific_name}), scientifically accurate, detailed feathers and anatomy, "
    "John Gould style, museum ornithology collection, "
    "pure white background, isolated bird, no text, no watermark, "
    "full body view, side profile"
)

NEGATIVE_PROMPT = (
    "multiple birds, flock, background scenery, landscape, text, "
    "watermark, signature, logo, frame, border, dark background, "
    "blurry, low quality, cartoon, anime"
)


class GeneratedArtworkProvider(ArtworkProvider):
    """
    Generates bird illustrations using ComfyUI + Flux.1 Dev GGUF,
    removes the background with rembg, and caches the results locally.

    Parameters
    ----------
    comfyui_url : str
        Base URL of the ComfyUI API. Default: http://127.0.0.1:8188
    cache_dir : Path | None
        Where to cache generated PNGs. Default: assets/artwork/generated/
    comfyui_output_dir : Path | None
        ComfyUI's output directory. Default: auto-detected from portable path.
    steps : int
        Sampling steps (30 = good quality, 4 = fast draft).
    width, height : int
        Output image dimensions.
    force_regenerate : bool
        If True, bypass cache and always regenerate.
    """

    def __init__(
        self,
        comfyui_url: str = COMFYUI_URL,
        cache_dir: Optional[Path] = None,
        comfyui_output_dir: Optional[Path] = None,
        steps: int = 30,
        width: int = 768,
        height: int = 768,
        force_regenerate: bool = False,
    ) -> None:
        self._url = comfyui_url.rstrip("/")
        self._cache_dir = cache_dir or GENERATED_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._comfyui_output = comfyui_output_dir or COMFYUI_OUTPUT_DIR
        self._steps = steps
        self._width = width
        self._height = height
        self._force_regenerate = force_regenerate

        logger.info(
            "GeneratedArtworkProvider: url=%s cache=%s steps=%d size=%dx%d",
            self._url, self._cache_dir, steps, width, height,
        )

    # ------------------------------------------------------------------
    # ArtworkProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "GeneratedArtworkProvider"

    @property
    def supported_species(self) -> list[str]:
        return [
            p.stem.replace("_", " ")
            for p in self._cache_dir.glob("*.png")
        ]

    def get_artwork(
        self,
        scientific_name: str,
        common_name: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Return the path to a cut-out illustration for *scientific_name*.

        Generates and caches on first call. Returns cached path on
        subsequent calls. Returns None if generation fails.

        Parameters
        ----------
        scientific_name : str
            e.g. "Erithacus rubecula"
        common_name : str | None
            e.g. "European Robin". If None, derived from scientific_name.
        """
        if not scientific_name or not scientific_name.strip():
            raise ArtworkProviderError("scientific_name must not be empty.")

        stem = scientific_name.lower().strip().replace(" ", "_")
        cache_path = self._cache_dir / f"{stem}.png"

        if not self._force_regenerate and cache_path.exists():
            logger.debug("Cache hit for %s: %s", scientific_name, cache_path)
            return cache_path

        # Derive common name if not provided
        if common_name is None:
            common_name = scientific_name.replace("_", " ").title()

        logger.info("Generating artwork for: %s (%s)", scientific_name, common_name)

        try:
            raw_path = self._generate_via_comfyui(
                scientific_name, common_name, stem
            )
        except ArtworkProviderError:
            raise
        except Exception as exc:
            logger.error("Generation failed for %s: %s", scientific_name, exc)
            return None

        if raw_path is None:
            return None

        # Remove background
        try:
            final_path = self._remove_background(raw_path, cache_path)
        except Exception as exc:
            logger.warning(
                "Background removal failed for %s (%s) — "
                "using raw image instead.",
                scientific_name, exc,
            )
            import shutil
            shutil.copy(raw_path, cache_path)
            final_path = cache_path

        logger.info("Artwork ready: %s", final_path)
        return final_path

    def has_artwork(self, scientific_name: str) -> bool:
        if not scientific_name or not scientific_name.strip():
            return False
        stem = scientific_name.lower().strip().replace(" ", "_")
        return (self._cache_dir / f"{stem}.png").exists()

    # ------------------------------------------------------------------
    # ComfyUI HTTP API
    # ------------------------------------------------------------------

    def _generate_via_comfyui(
        self,
        scientific_name: str,
        common_name: str,
        stem: str,
    ) -> Optional[Path]:
        """
        Submit a workflow to ComfyUI and wait for the output image.

        Returns the local Path to the downloaded raw image, or None.
        """
        # Check ComfyUI is reachable
        if not self._ping_comfyui():
            raise ArtworkProviderError(
                f"ComfyUI is not reachable at {self._url}. "
                "Make sure ComfyUI is running."
            )

        # Build workflow with substituted prompt
        workflow = self._build_workflow(scientific_name, common_name, stem)
        client_id = str(uuid.uuid4())

        # Submit
        prompt_id = self._submit_workflow(workflow, client_id)
        if prompt_id is None:
            raise ArtworkProviderError("ComfyUI did not accept the workflow.")

        logger.debug("Submitted workflow prompt_id=%s", prompt_id)

        # Poll until done
        output_filename = self._wait_for_completion(prompt_id)
        if output_filename is None:
            raise ArtworkProviderError(
                f"ComfyUI generation timed out after {GENERATION_TIMEOUT}s."
            )

        # Download / copy the image
        raw_path = self._retrieve_image(output_filename, stem)
        return raw_path

    def _ping_comfyui(self) -> bool:
        """Return True if ComfyUI API is reachable."""
        try:
            req = urllib.request.Request(f"{self._url}/system_stats")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    def _build_workflow(
        self,
        scientific_name: str,
        common_name: str,
        stem: str,
    ) -> dict:
        """Build the ComfyUI workflow dict with substituted prompts."""
        positive = PROMPT_TEMPLATE.format(
            common_name=common_name,
            scientific_name=scientific_name,
        )
        filename_prefix = f"birdframe_{stem}"

        return {
            "prompt": {
                "1": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {"unet_name": "flux1-dev-Q4_K_S.gguf"},
                },
                "10": {
                    "class_type": "DualCLIPLoader",
                    "inputs": {
                        "clip_name1": "clip_l.safetensors",
                        "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                        "type": "flux",
                    },
                },
                "9": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": "ae.safetensors"},
                },
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "clip": ["10", 0],
                        "text": positive,
                    },
                },
                "3": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "clip": ["10", 0],
                        "text": NEGATIVE_PROMPT,
                    },
                },
                "4": {
                    "class_type": "EmptySD3LatentImage",
                    "inputs": {
                        "width": self._width,
                        "height": self._height,
                        "batch_size": 1,
                    },
                },
                "5": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["1", 0],
                        "positive": ["2", 0],
                        "negative": ["3", 0],
                        "latent_image": ["4", 0],
                        "seed": int(time.time()) % 2**32,
                        "steps": self._steps,
                        "cfg": 1.0,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 1.0,
                    },
                },
                "6": {
                    "class_type": "VAEDecode",
                    "inputs": {
                        "samples": ["5", 0],
                        "vae": ["9", 0],
                    },
                },
                "7": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["6", 0],
                        "filename_prefix": filename_prefix,
                    },
                },
            }
        }

    def _submit_workflow(self, workflow: dict, client_id: str) -> Optional[str]:
        """POST the workflow to ComfyUI and return the prompt_id."""
        payload = json.dumps({
            "prompt": workflow["prompt"],
            "client_id": client_id,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("prompt_id")
        except Exception as exc:
            logger.error("Failed to submit workflow: %s", exc)
            return None

    def _wait_for_completion(self, prompt_id: str) -> Optional[str]:
        """
        Poll ComfyUI history until the prompt is done.
        Returns the output filename or None on timeout.
        """
        deadline = time.time() + GENERATION_TIMEOUT
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            try:
                req = urllib.request.Request(
                    f"{self._url}/history/{prompt_id}"
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    history = json.loads(resp.read())

                if prompt_id not in history:
                    continue

                prompt_data = history[prompt_id]

                # Check for errors
                status = prompt_data.get("status", {})
                if status.get("status_str") == "error":
                    logger.error(
                        "ComfyUI reported error for prompt %s", prompt_id
                    )
                    return None

                # Look for output images
                outputs = prompt_data.get("outputs", {})
                for node_id, node_output in outputs.items():
                    images = node_output.get("images", [])
                    if images:
                        filename = images[0]["filename"]
                        logger.debug(
                            "Generation complete: %s", filename
                        )
                        return filename

            except Exception as exc:
                logger.debug("Poll error (will retry): %s", exc)

        return None  # timed out

    def _retrieve_image(
        self, filename: str, stem: str
    ) -> Optional[Path]:
        """
        Try to get the generated image:
        1. Copy from local ComfyUI output directory (fastest)
        2. Download via ComfyUI /view API (fallback)
        """
        raw_path = self._cache_dir / f"{stem}_raw.png"

        # Option 1: copy locally
        local = self._comfyui_output / filename
        if local.exists():
            import shutil
            shutil.copy(local, raw_path)
            logger.debug("Copied from local output: %s", local)
            return raw_path

        # Option 2: download via API
        url = f"{self._url}/view?filename={filename}&type=output"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_path.write_bytes(resp.read())
            logger.debug("Downloaded via API: %s", filename)
            return raw_path
        except Exception as exc:
            logger.error("Could not retrieve image %s: %s", filename, exc)
            return None

    # ------------------------------------------------------------------
    # Background removal
    # ------------------------------------------------------------------

    def _remove_background(self, raw_path: Path, out_path: Path) -> Path:
        """
        Remove the white/parchment background from raw_path using rembg.
        Saves the result as a transparent PNG to out_path.
        """
        try:
            from rembg import remove
            from PIL import Image
        except ImportError as exc:
            raise ArtworkProviderError(
                "rembg is not installed. Run: pip install rembg"
            ) from exc

        logger.debug("Removing background from %s", raw_path)
        with Image.open(str(raw_path)) as img:
            # Convert to RGBA before processing
            img_rgba = img.convert("RGBA")
            result = remove(img_rgba)
            result.save(str(out_path), "PNG")

        # Clean up raw file
        try:
            raw_path.unlink()
        except OSError:
            pass

        return out_path

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_cache(self, scientific_name: Optional[str] = None) -> int:
        """
        Delete cached artwork.

        If scientific_name is provided, delete only that species.
        If None, delete all cached artwork.
        Returns the number of files deleted.
        """
        if scientific_name:
            stem = scientific_name.lower().strip().replace(" ", "_")
            paths = list(self._cache_dir.glob(f"{stem}*.png"))
        else:
            paths = list(self._cache_dir.glob("*.png"))

        for p in paths:
            p.unlink(missing_ok=True)

        logger.info("Cleared %d cached artwork file(s).", len(paths))
        return len(paths)
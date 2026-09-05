"""
CollageGenerator — composes a "Heard Recently" image from bird artwork.

Takes a list of species (with artwork paths) and arranges them into a
single image suitable for display on a screen or frame.

Layout: a clean grid of bird illustrations, each with the species name
below it. Background is off-white parchment to complement the vintage
natural history aesthetic.

The generator is stateless — call generate() as many times as needed.
Output is a JPEG saved to config.COLLAGE_DIR.

Usage:

    from backend.collage.generator import CollageGenerator
    from backend.artwork.static_provider import StaticArtworkProvider
    from backend.database.repository import DetectionRepository

    generator = CollageGenerator()
    provider = StaticArtworkProvider()

    species_paths = {
        "Erithacus rubecula": ("European Robin", provider.get_artwork("Erithacus rubecula")),
        "Parus major": ("Great Tit", provider.get_artwork("Parus major")),
    }

    output_path = generator.generate(species_paths)
    print(f"Collage saved to: {output_path}")
"""

import sys
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config

logger = logging.getLogger(__name__)

# Parchment/paper colour — warm off-white to complement vintage illustrations
BACKGROUND_COLOUR = (245, 240, 228)

# Text colours
TITLE_COLOUR = (60, 45, 30)       # dark brown
SPECIES_COLOUR = (80, 60, 40)     # medium brown
SUBTITLE_COLOUR = (120, 100, 80)  # muted brown

# Padding and spacing
CELL_PADDING = 30          # pixels inside each grid cell
LABEL_HEIGHT = 60          # reserved for species name below each image
TITLE_AREA_HEIGHT = 100    # reserved for the "Heard Recently" title


class CollageGeneratorError(Exception):
    """Raised when the CollageGenerator cannot produce an image."""
    pass


class CollageGenerator:
    """
    Composes a "Heard Recently" collage image from bird artwork.

    Parameters
    ----------
    width : int
        Output image width in pixels. Defaults to config.COLLAGE_WIDTH.
    height : int
        Output image height in pixels. Defaults to config.COLLAGE_HEIGHT.
    output_dir : Path | None
        Directory to save generated collages. Defaults to config.COLLAGE_DIR.
    max_species : int
        Maximum number of species to include. Defaults to config.COLLAGE_MAX_SPECIES.
    title : str
        Text shown at the top of the collage.
    """

    def __init__(
        self,
        width: int = config.COLLAGE_WIDTH,
        height: int = config.COLLAGE_HEIGHT,
        output_dir: Optional[Path] = None,
        max_species: int = config.COLLAGE_MAX_SPECIES,
        title: str = "Heard Recently",
    ) -> None:
        self._width = width
        self._height = height
        self._output_dir = output_dir or config.COLLAGE_DIR
        self._max_species = max_species
        self._title = title

        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def generate(
        self,
        species_paths: dict[str, tuple[str, Optional[Path]]],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate a collage image and save it to output_dir.

        Parameters
        ----------
        species_paths : dict[str, tuple[str, Optional[Path]]]
            Mapping of scientific_name → (common_name, artwork_path).
            artwork_path may be None — those species are skipped.
        filename : str | None
            Output filename. Defaults to collage_<timestamp>.jpg.

        Returns
        -------
        Path
            Absolute path to the saved collage image.

        Raises
        ------
        CollageGeneratorError
            If Pillow is not installed or no valid artwork is available.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise CollageGeneratorError(
                "Pillow is not installed. Run: pip install pillow"
            ) from exc

        # Filter to species that have artwork
        valid = {
            sci: (common, path)
            for sci, (common, path) in species_paths.items()
            if path is not None and path.exists()
        }

        if not valid:
            raise CollageGeneratorError(
                "No valid artwork paths provided — cannot generate collage."
            )

        # Limit to max_species
        if len(valid) > self._max_species:
            valid = dict(list(valid.items())[: self._max_species])

        logger.info(
            "Generating collage: %d species, %dx%d px",
            len(valid), self._width, self._height,
        )

        # Build the image
        img = self._build_image(valid, Image, ImageDraw, ImageFont)

        # Save
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"collage_{ts}.jpg"

        output_path = self._output_dir / filename
        img.save(str(output_path), "JPEG", quality=92)
        logger.info("Collage saved: %s", output_path)

        return output_path

    def generate_latest(
        self,
        species_paths: dict[str, tuple[str, Optional[Path]]],
    ) -> Path:
        """
        Generate a collage and save it as 'latest.jpg', overwriting any
        previous version. Useful for the API endpoint that always serves
        the most recent collage.
        """
        return self.generate(species_paths, filename="latest.jpg")

    # ------------------------------------------------------------------
    # Internal image construction
    # ------------------------------------------------------------------

    def _build_image(self, valid, Image, ImageDraw, ImageFont):
        """Compose the full collage image and return a PIL Image."""
        img = Image.new("RGB", (self._width, self._height), BACKGROUND_COLOUR)
        draw = ImageDraw.Draw(img)

        # Draw title
        self._draw_title(draw, ImageFont)

        # Compute grid layout
        n = len(valid)
        cols, rows = self._grid_dimensions(n)
        cell_w = self._width // cols
        cell_h = (self._height - TITLE_AREA_HEIGHT) // rows

        # Draw each species cell
        for idx, (sci, (common, path)) in enumerate(valid.items()):
            row = idx // cols
            col = idx % cols
            x = col * cell_w
            y = TITLE_AREA_HEIGHT + row * cell_h
            self._draw_cell(img, draw, ImageFont, x, y, cell_w, cell_h,
                            sci, common, path, Image)

        # Draw subtle border
        draw.rectangle(
            [2, 2, self._width - 3, self._height - 3],
            outline=(180, 160, 130),
            width=3,
        )

        return img

    def _draw_title(self, draw, ImageFont) -> None:
        """Draw the title text at the top of the image."""
        font = self._get_font(ImageFont, size=52, bold=True)
        subtitle_font = self._get_font(ImageFont, size=22)

        # Main title
        title_bbox = draw.textbbox((0, 0), self._title, font=font)
        title_w = title_bbox[2] - title_bbox[0]
        title_x = (self._width - title_w) // 2
        draw.text((title_x, 18), self._title, fill=TITLE_COLOUR, font=font)

        # Subtitle with timestamp
        now_str = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
        sub_bbox = draw.textbbox((0, 0), now_str, font=subtitle_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_x = (self._width - sub_w) // 2
        draw.text((sub_x, 72), now_str, fill=SUBTITLE_COLOUR, font=subtitle_font)

        # Separator line
        draw.line(
            [(40, TITLE_AREA_HEIGHT - 8), (self._width - 40, TITLE_AREA_HEIGHT - 8)],
            fill=(180, 160, 130),
            width=2,
        )

    def _draw_cell(
        self, img, draw, ImageFont,
        x, y, cell_w, cell_h,
        sci, common, path, Image,
    ) -> None:
        """Draw one species: artwork + name label, within its grid cell."""
        # Image area (leaves room for label below)
        img_area_h = cell_h - LABEL_HEIGHT
        img_x = x + CELL_PADDING
        img_y = y + CELL_PADDING
        img_w = cell_w - 2 * CELL_PADDING
        img_h = img_area_h - CELL_PADDING

        if img_w <= 0 or img_h <= 0:
            return

        try:
            bird_img = Image.open(str(path)).convert("RGB")
            bird_img = self._fit_image(bird_img, img_w, img_h)

            # Centre within the cell's image area
            offset_x = img_x + (img_w - bird_img.width) // 2
            offset_y = img_y + (img_h - bird_img.height) // 2
            img.paste(bird_img, (offset_x, offset_y))

        except Exception as exc:
            logger.warning("Could not load artwork %s: %s", path, exc)
            # Draw a placeholder rectangle
            draw.rectangle(
                [img_x, img_y, img_x + img_w, img_y + img_h],
                outline=(180, 160, 130),
                width=2,
            )

        # Species name label below the image
        label_y = y + cell_h - LABEL_HEIGHT + 8
        self._draw_species_label(draw, ImageFont, x, label_y, cell_w, common, sci)

    def _draw_species_label(
        self, draw, ImageFont,
        cell_x, label_y, cell_w,
        common_name: str,
        scientific_name: str,
    ) -> None:
        """Draw common name and scientific name below the artwork."""
        common_font = self._get_font(ImageFont, size=18, bold=True)
        sci_font = self._get_font(ImageFont, size=14, italic=True)

        # Common name — centred
        cb = draw.textbbox((0, 0), common_name, font=common_font)
        cw = cb[2] - cb[0]
        cx = cell_x + (cell_w - cw) // 2
        draw.text((cx, label_y), common_name, fill=SPECIES_COLOUR, font=common_font)

        # Scientific name — centred below
        sb = draw.textbbox((0, 0), scientific_name, font=sci_font)
        sw = sb[2] - sb[0]
        sx = cell_x + (cell_w - sw) // 2
        draw.text((sx, label_y + 24), scientific_name,
                  fill=SUBTITLE_COLOUR, font=sci_font)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_dimensions(n: int) -> tuple[int, int]:
        """
        Return (cols, rows) for a grid that fits n items.

        Prefers wider layouts (more columns than rows) to suit
        landscape display orientation.
        """
        if n <= 0:
            return 1, 1
        if n == 1:
            return 1, 1
        if n == 2:
            return 2, 1
        if n == 3:
            return 3, 1
        if n == 4:
            return 2, 2
        if n <= 6:
            return 3, 2
        if n <= 8:
            return 4, 2
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return cols, rows

    @staticmethod
    def _fit_image(img, max_w: int, max_h: int):
        """Resize img to fit within max_w × max_h, preserving aspect ratio."""
        iw, ih = img.size
        if iw == 0 or ih == 0:
            return img
        scale = min(max_w / iw, max_h / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        from PIL import Image
        return img.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def _get_font(ImageFont, size: int = 16,
                  bold: bool = False, italic: bool = False):
        """
        Return a PIL ImageFont. Falls back to the built-in default font
        if no system fonts are available (which is always safe).
        """
        # Try common system font names in order of preference
        candidates = []
        if bold and italic:
            candidates = [
                "georgiabi.ttf", "timesbi.ttf", "trebucbi.ttf",
            ]
        elif bold:
            candidates = [
                "georgiab.ttf", "timesbd.ttf", "trebucbd.ttf",
                "arialbd.ttf", "calibrib.ttf",
            ]
        elif italic:
            candidates = [
                "georgiai.ttf", "timesi.ttf", "trebucit.ttf",
                "ariali.ttf",
            ]
        else:
            candidates = [
                "georgia.ttf", "times.ttf", "trebuc.ttf",
                "arial.ttf", "calibri.ttf",
            ]

        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue

        # Final fallback: PIL default bitmap font (always available)
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            # Older Pillow versions don't accept size
            return ImageFont.load_default()
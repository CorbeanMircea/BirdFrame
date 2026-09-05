"""
CollageGenerator — composes a "Heard Recently" scattered collage.

Layout: bird illustrations are placed freely across the canvas at
varying sizes and slight rotations. The placement algorithm actively
avoids overlap by trying many candidate positions and picking the
one with the least overlap.

Output is a JPEG saved to config.COLLAGE_DIR.
"""

import sys
import logging
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config

logger = logging.getLogger(__name__)

# Parchment background
BACKGROUND_COLOUR = (245, 240, 228)

# Text colours
TITLE_COLOUR   = (60, 45, 30)
SUBTITLE_COLOUR = (130, 110, 85)

# Layout parameters
MIN_BIRD_SCALE    = 0.12   # smallest bird as fraction of canvas width
MAX_BIRD_SCALE    = 0.20   # largest bird as fraction of canvas width
MAX_ROTATION      = 18     # degrees either side
PLACEMENT_TRIES   = 200    # candidate positions tried per bird
MARGIN            = 20     # minimum pixels from canvas edge
FOOTER_HEIGHT     = 70     # reserved at the bottom for title bar
# Minimum clearance between birds (pixels). Set to 0 to allow touching.
MIN_GAP           = 15


class CollageGeneratorError(Exception):
    pass


class CollageGenerator:
    """
    Composes a scattered "Heard Recently" collage image.

    Parameters
    ----------
    width, height : int
        Output dimensions in pixels.
    output_dir : Path | None
        Where to save generated images.
    max_species : int
        Cap on species shown.
    title : str
        Footer title text.
    seed : int | None
        Fix the random seed for a reproducible layout.
    """

    def __init__(
        self,
        width: int = config.COLLAGE_WIDTH,
        height: int = config.COLLAGE_HEIGHT,
        output_dir: Optional[Path] = None,
        max_species: int = config.COLLAGE_MAX_SPECIES,
        title: str = "Heard Recently",
        seed: Optional[int] = None,
    ) -> None:
        self._width = width
        self._height = height
        self._output_dir = output_dir or config.COLLAGE_DIR
        self._max_species = max_species
        self._title = title
        self._seed = seed
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        species_paths: dict[str, tuple[str, Optional[Path]]],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate a scattered collage and save it to output_dir.

        Parameters
        ----------
        species_paths : dict[str, tuple[str, Optional[Path]]]
            scientific_name → (common_name, artwork_path | None)
        filename : str | None
            Output filename. Defaults to collage_<timestamp>.jpg.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
        except ImportError as exc:
            raise CollageGeneratorError(
                "Pillow is not installed. Run: pip install pillow"
            ) from exc

        valid = {
            sci: (common, path)
            for sci, (common, path) in species_paths.items()
            if path is not None and Path(path).exists()
        }
        if not valid:
            raise CollageGeneratorError(
                "No valid artwork paths — cannot generate collage."
            )
        if len(valid) > self._max_species:
            valid = dict(list(valid.items())[: self._max_species])

        rng = random.Random(self._seed)

        canvas = Image.new("RGB", (self._width, self._height), BACKGROUND_COLOUR)
        canvas = self._add_paper_texture(canvas, rng, Image)

        self._scatter_birds(canvas, valid, rng, Image)
        self._draw_footer(canvas, ImageDraw, ImageFont)
        self._draw_border(canvas, ImageDraw)

        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"collage_{ts}.jpg"

        output_path = self._output_dir / filename
        canvas.save(str(output_path), "JPEG", quality=93)
        logger.info("Collage saved: %s (%d species)", output_path, len(valid))
        return output_path

    def generate_latest(
        self,
        species_paths: dict[str, tuple[str, Optional[Path]]],
    ) -> Path:
        """Generate and save as latest.jpg, overwriting any previous version."""
        return self.generate(species_paths, filename="latest.jpg")

    # ------------------------------------------------------------------
    # Paper texture
    # ------------------------------------------------------------------

    def _add_paper_texture(self, canvas, rng, Image):
        """Subtle noise grain to simulate aged paper."""
        try:
            import numpy as np
            w, h = canvas.size
            np.random.seed(rng.randint(0, 0xFFFF))
            grain = np.random.randint(-10, 11, (h, w, 3), dtype=np.int16)
            arr = np.clip(np.array(canvas, dtype=np.int16) + grain, 0, 255)
            return Image.fromarray(arr.astype(np.uint8))
        except ImportError:
            return canvas

    # ------------------------------------------------------------------
    # Bird scattering — overlap-aware placement
    # ------------------------------------------------------------------

    def _scatter_birds(self, canvas, valid, rng, Image) -> None:
        """
        Place bird illustrations across the canvas.

        For each bird we try PLACEMENT_TRIES random positions and pick
        the one with zero overlap if possible, or minimum overlap if not.
        Birds are sorted largest-first so big birds are placed first,
        making it easier to slot smaller ones around them.
        """
        items = list(valid.items())
        rng.shuffle(items)

        n = len(items)
        scales = self._pick_scales(n, rng)

        # Pre-load and resize all images so we know their sizes before placing
        prepared: list[tuple[str, str, object, float]] = []
        for (sci, (common, path)), scale in zip(items, scales):
            try:
                bird = Image.open(str(path)).convert("RGBA")
            except Exception as exc:
                logger.warning("Cannot open %s: %s", path, exc)
                continue

            target_w = int(self._width * scale)
            bird = _fit_to_width(bird, target_w)

            angle = rng.uniform(-MAX_ROTATION, MAX_ROTATION)
            if rng.random() < 0.45:
                bird = bird.transpose(Image.FLIP_LEFT_RIGHT)
            bird = bird.rotate(angle, expand=True, resample=Image.BICUBIC)

            prepared.append((sci, common, bird, scale))

        # Sort largest first (by area) so big birds are placed first
        prepared.sort(key=lambda t: t[2].size[0] * t[2].size[1], reverse=True)

        # Canvas area available for placement (excluding footer)
        available_h = self._height - FOOTER_HEIGHT - MARGIN

        # Placed bounding boxes with a gap buffer
        placed: list[tuple[int, int, int, int]] = []

        for sci, common, bird_img, scale in prepared:
            bw, bh = bird_img.size

            x, y = self._best_placement(bw, bh, placed, available_h, rng)

            # Paste with alpha mask
            mask = self._make_mask(bird_img, Image)
            canvas.paste(bird_img.convert("RGB"), (x, y), mask)

            # Record placed bbox (with gap buffer for next placements)
            placed.append((
                x - MIN_GAP,
                y - MIN_GAP,
                x + bw + MIN_GAP,
                y + bh + MIN_GAP,
            ))

    def _best_placement(
        self,
        bw: int,
        bh: int,
        placed: list[tuple[int, int, int, int]],
        available_h: int,
        rng: random.Random,
    ) -> tuple[int, int]:
        """
        Try PLACEMENT_TRIES random positions and return the best one.

        'Best' = zero overlap if achievable, otherwise minimum total
        overlap area. We sample positions from a grid-jittered
        distribution so candidates cover the canvas more evenly.
        """
        max_x = max(MARGIN, self._width - bw - MARGIN)
        max_y = max(MARGIN, available_h - bh - MARGIN)

        # Generate candidate positions using grid jitter for even coverage
        candidates = _generate_candidates(
            max_x, max_y, PLACEMENT_TRIES, rng
        )

        best_pos = (rng.randint(MARGIN, max(MARGIN + 1, max_x)),
                    rng.randint(MARGIN, max(MARGIN + 1, max_y)))
        best_overlap = float("inf")

        for cx, cy in candidates:
            # Clamp to valid range
            cx = max(MARGIN, min(cx, max_x))
            cy = max(MARGIN, min(cy, max_y))

            ov = _total_overlap(cx, cy, bw, bh, placed)
            if ov < best_overlap:
                best_overlap = ov
                best_pos = (cx, cy)
            if ov == 0:
                break  # perfect placement found — stop early

        return best_pos

    def _pick_scales(self, n: int, rng: random.Random) -> list[float]:
        """Mix of larger feature birds and smaller background birds."""
        scales = []
        for i in range(n):
            if i == 0:
                # One prominent feature bird
                s = rng.uniform(MAX_BIRD_SCALE * 0.9, MAX_BIRD_SCALE)
            elif i < max(2, n // 3):
                s = rng.uniform(MAX_BIRD_SCALE * 0.65, MAX_BIRD_SCALE * 0.85)
            else:
                s = rng.uniform(MIN_BIRD_SCALE, MAX_BIRD_SCALE * 0.65)
            scales.append(s)
        rng.shuffle(scales)
        return scales

    @staticmethod
    def _make_mask(img, Image):
        if img.mode == "RGBA":
            return img.split()[3]
        return Image.new("L", img.size, 255)

    # ------------------------------------------------------------------
    # Footer and border
    # ------------------------------------------------------------------

    def _draw_footer(self, canvas, ImageDraw, ImageFont) -> None:
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        footer_y = h - FOOTER_HEIGHT

        draw.rectangle([0, footer_y, w, h], fill=(235, 228, 210))
        draw.line([(0, footer_y), (w, footer_y)], fill=(180, 160, 130), width=2)

        title_font = self._get_font(ImageFont, size=32, bold=True)
        sub_font = self._get_font(ImageFont, size=18)

        draw.text((30, footer_y + 10), self._title,
                  fill=TITLE_COLOUR, font=title_font)

        now_str = datetime.now(timezone.utc).strftime("%d %B %Y  %H:%M UTC")
        sub_bbox = draw.textbbox((0, 0), now_str, font=sub_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        draw.text((w - sub_w - 30, footer_y + 22), now_str,
                  fill=SUBTITLE_COLOUR, font=sub_font)

    def _draw_border(self, canvas, ImageDraw) -> None:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            [3, 3, self._width - 4, self._height - 4],
            outline=(180, 160, 130),
            width=3,
        )

    # ------------------------------------------------------------------
    # Grid layout (kept for tests and fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_dimensions(n: int) -> tuple[int, int]:
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
        candidates = []
        if bold:
            candidates = ["georgiab.ttf", "timesbd.ttf", "trebucbd.ttf",
                          "arialbd.ttf", "calibrib.ttf"]
        elif italic:
            candidates = ["georgiai.ttf", "timesi.ttf", "ariali.ttf"]
        else:
            candidates = ["georgia.ttf", "times.ttf", "trebuc.ttf",
                          "arial.ttf", "calibri.ttf"]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _fit_to_width(img, target_w: int):
    from PIL import Image
    iw, ih = img.size
    if iw == 0:
        return img
    scale = target_w / iw
    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _generate_candidates(
    max_x: int,
    max_y: int,
    n: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """
    Generate n candidate (x, y) positions using grid jitter.

    Divides the canvas into a grid and samples one jittered point
    per cell, ensuring candidates are spread across the full canvas
    rather than clustered in one area.
    """
    if max_x <= 0 or max_y <= 0:
        return [(0, 0)] * n

    # Grid dimensions
    cols = max(1, int(math.sqrt(n * max_x / max(1, max_y))))
    rows = max(1, math.ceil(n / cols))
    cell_w = max_x // cols
    cell_h = max_y // rows

    candidates = []
    for row in range(rows):
        for col in range(cols):
            if len(candidates) >= n:
                break
            base_x = col * cell_w
            base_y = row * cell_h
            jx = rng.randint(0, max(0, cell_w - 1))
            jy = rng.randint(0, max(0, cell_h - 1))
            candidates.append((base_x + jx, base_y + jy))

    # Top up with pure random candidates if grid didn't produce enough
    while len(candidates) < n:
        candidates.append((
            rng.randint(0, max_x),
            rng.randint(0, max_y),
        ))

    rng.shuffle(candidates)
    return candidates[:n]


def _overlap_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> int:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _total_overlap(
    x: int, y: int, w: int, h: int,
    placed: list[tuple[int, int, int, int]],
) -> int:
    return sum(
        _overlap_area(x, y, x + w, y + h, px1, py1, px2, py2)
        for px1, py1, px2, py2 in placed
    )
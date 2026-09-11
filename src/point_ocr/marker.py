"""Frozen crosshair marker protocol.

Training and inference MUST use the same MARKER_SPEC so the model learns a
stable visual cue that survives resize / compression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class MarkerSpec:
    """Geometry and colors for the point-of-interest crosshair."""

    # Outer white ring (high-contrast halo)
    ring_radius_px: int = 28
    ring_width_px: int = 5
    ring_color: tuple[int, int, int, int] = (255, 255, 255, 255)

    # Magenta cross arms
    cross_half_length_px: int = 34
    cross_width_px: int = 5
    cross_color: tuple[int, int, int, int] = (255, 0, 255, 255)  # #FF00FF

    # Inner white outline on cross (helps on dark/busy backgrounds)
    cross_outline_width_px: int = 2
    cross_outline_color: tuple[int, int, int, int] = (255, 255, 255, 255)

    # Dot at center
    center_dot_radius_px: int = 4
    center_dot_color: tuple[int, int, int, int] = (255, 0, 255, 255)

    # Minimum rendered size after any resize (sanity floor for builders)
    min_visible_extent_px: int = 48

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Single source of truth — import this everywhere.
MARKER_SPEC = MarkerSpec()


def draw_crosshair(
    image: Image.Image,
    x: float,
    y: float,
    spec: MarkerSpec = MARKER_SPEC,
    *,
    copy: bool = True,
) -> Image.Image:
    """Overlay a magenta/white crosshair at (x, y) in pixel coordinates.

    Coordinates may be float; they are rounded to nearest pixel.
    Returns RGB image (alpha composited if needed).
    """
    base = image.convert("RGBA")
    if copy:
        base = base.copy()

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cx = int(round(x))
    cy = int(round(y))
    w, h = base.size
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    # Outer ring
    r = spec.ring_radius_px
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=spec.ring_color,
        width=spec.ring_width_px,
    )

    half = spec.cross_half_length_px
    # White outline under magenta arms
    ow = spec.cross_width_px + 2 * spec.cross_outline_width_px
    draw.line([(cx - half, cy), (cx + half, cy)], fill=spec.cross_outline_color, width=ow)
    draw.line([(cx, cy - half), (cx, cy + half)], fill=spec.cross_outline_color, width=ow)

    # Magenta arms
    draw.line([(cx - half, cy), (cx + half, cy)], fill=spec.cross_color, width=spec.cross_width_px)
    draw.line([(cx, cy - half), (cx, cy + half)], fill=spec.cross_color, width=spec.cross_width_px)

    # Center dot
    d = spec.center_dot_radius_px
    draw.ellipse([cx - d, cy - d, cx + d, cy + d], fill=spec.center_dot_color)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out


def marker_extent_ok(image_w: int, image_h: int, spec: MarkerSpec = MARKER_SPEC) -> bool:
    """True if the image is large enough for the marker to remain visible after mild resize."""
    extent = max(spec.ring_radius_px * 2, spec.cross_half_length_px * 2)
    return (
        image_w >= spec.min_visible_extent_px
        and image_h >= spec.min_visible_extent_px
        and extent <= min(image_w, image_h)
    )

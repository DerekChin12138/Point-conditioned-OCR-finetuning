"""Frozen crosshair marker protocol.

Training and inference MUST use the same MARKER_SPEC so the model learns a
stable visual cue that survives resize / compression.

Batch-2 geometry: slightly smaller overall footprint + open center (no solid
dot / gap in the cross arms) so short glyphs under the interest point stay
readable. Size is fixed — never scaled by bbox (product cannot know block size).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class MarkerSpec:
    """Geometry and colors for the point-of-interest crosshair."""

    # Outer white ring (high-contrast halo)
    ring_radius_px: int = 22
    ring_width_px: int = 4
    ring_color: tuple[int, int, int, int] = (255, 255, 255, 255)

    # Magenta cross arms (open-center: arms stop at center_gap_radius_px)
    cross_half_length_px: int = 26
    cross_width_px: int = 4
    cross_color: tuple[int, int, int, int] = (255, 0, 255, 255)  # #FF00FF

    # Inner white outline on cross (helps on dark/busy backgrounds)
    cross_outline_width_px: int = 2
    cross_outline_color: tuple[int, int, int, int] = (255, 255, 255, 255)

    # Clear aperture at the interest point (no fill). Arms do not enter this radius.
    center_gap_radius_px: int = 7

    # Center dot disabled (0). Kept for backward-compatible field presence.
    center_dot_radius_px: int = 0
    center_dot_color: tuple[int, int, int, int] = (255, 0, 255, 255)

    # Minimum rendered size after any resize (sanity floor for builders)
    min_visible_extent_px: int = 40

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Single source of truth — import this everywhere.
MARKER_SPEC = MarkerSpec()


def _draw_open_arm(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    half: int,
    gap: int,
    horizontal: bool,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    """Draw one cross axis as two segments leaving a clear center gap."""
    if half <= gap:
        return
    if horizontal:
        draw.line([(cx - half, cy), (cx - gap, cy)], fill=fill, width=width)
        draw.line([(cx + gap, cy), (cx + half, cy)], fill=fill, width=width)
    else:
        draw.line([(cx, cy - half), (cx, cy - gap)], fill=fill, width=width)
        draw.line([(cx, cy + gap), (cx, cy + half)], fill=fill, width=width)


def draw_crosshair(
    image: Image.Image,
    x: float,
    y: float,
    spec: MarkerSpec = MARKER_SPEC,
    *,
    copy: bool = True,
) -> Image.Image:
    """Overlay a magenta/white open-center crosshair at (x, y) in pixel coordinates.

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
    gap = max(0, int(spec.center_gap_radius_px))
    # White outline under magenta arms
    ow = spec.cross_width_px + 2 * spec.cross_outline_width_px
    _draw_open_arm(
        draw,
        cx=cx,
        cy=cy,
        half=half,
        gap=gap,
        horizontal=True,
        fill=spec.cross_outline_color,
        width=ow,
    )
    _draw_open_arm(
        draw,
        cx=cx,
        cy=cy,
        half=half,
        gap=gap,
        horizontal=False,
        fill=spec.cross_outline_color,
        width=ow,
    )

    # Magenta arms
    _draw_open_arm(
        draw,
        cx=cx,
        cy=cy,
        half=half,
        gap=gap,
        horizontal=True,
        fill=spec.cross_color,
        width=spec.cross_width_px,
    )
    _draw_open_arm(
        draw,
        cx=cx,
        cy=cy,
        half=half,
        gap=gap,
        horizontal=False,
        fill=spec.cross_color,
        width=spec.cross_width_px,
    )

    # Optional center dot (Batch-2 default: radius 0 → skipped)
    d = spec.center_dot_radius_px
    if d > 0:
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

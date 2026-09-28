"""Frozen marker protocols.

Training and inference MUST use the same marker drawer for a given stage so the
model learns a stable visual cue that survives resize / compression.

- **Current: ``spec_for_image`` + ``draw_crosshair``** — magenta/white X (45°)
  whose bounding square (X = square diagonals) is a constant fraction of the
  page area. Opacity falls as that square grows so a large X still shows glyphs.
- A2 legacy: ``MARKER_SPEC_A2`` + ``draw_scatter_star`` (dotted four-pointed star)
- V2 legacy: ``MARKER_SPEC_V2`` + ``draw_translucent_dot`` (adaptive circle)

``MARKER_SPEC`` is the 1× geometry *template* (arm half-length 34px). Production
stamps must call ``spec_for_image(w, h)`` so size tracks page area.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class MarkerSpec:
    """Magenta/white interest-point cross (1× geometry template).

    Arms are the successful A1 geometry, rotated ``rotation_deg`` so a 45° X
    cuts between horizontal text strokes. Pixel size and alpha on a given
    page come from ``spec_for_image``, not from these template defaults.
    """

    ring_radius_px: int = 28
    ring_width_px: int = 3
    ring_color: tuple[int, int, int, int] = (255, 255, 255, 255)

    cross_half_length_px: int = 34
    cross_width_px: int = 3
    cross_color: tuple[int, int, int, int] = (255, 0, 255, 255)

    cross_outline_width_px: int = 1
    cross_outline_color: tuple[int, int, int, int] = (255, 255, 255, 255)

    center_dot_radius_px: int = 3
    center_dot_color: tuple[int, int, int, int] = (255, 0, 255, 255)

    rotation_deg: float = 45.0

    min_visible_extent_px: int = 48

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScatterStarSpec:
    """A2 legacy marker: four-pointed star made of sparse round dots."""

    arm_half_length_px: int = 30
    dot_spacing_px: float = 10.0
    dot_radius_px: int = 2
    dot_color: tuple[int, int, int, int] = (255, 0, 255, 150)
    outline_color: tuple[int, int, int, int] = (255, 255, 255, 130)
    outline_radius_px: int = 3
    draw_center_dot: bool = True
    ring_radius_px: int = 26
    ring_dot_count: int = 8
    min_visible_extent_px: int = 48

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DotMarkerSpec:
    """Current protocol: semi-opaque adaptive circle + solid outline (ViT-visible).

    Diameter must cover several vision patches; fill stays slightly translucent
    so glyphs under the mark remain partly readable. No extra center dot —
    a uniform disk (fill + outline) is the whole cue.
    """

    radius_px: int = 20
    fill_alpha: int = 200
    outline_width_px: int = 3
    outline_alpha: int = 255
    sample_patch_px: int = 36
    min_visible_extent_px: int = 48

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MARKER_SPEC = MarkerSpec()
MARKER_SPEC_A2 = ScatterStarSpec()
MARKER_SPEC_V2 = DotMarkerSpec()

# x45r: same X geometry, but each marked image draws its page-area fraction
# uniformly from [MIN, MAX].  Marker-size variation is carried by the SFT data,
# so it is no longer something GRPO has to learn.
CURRENT_MARKER_TAG = "x45r"
MARKER_AREA_FRAC_MIN: float = 0.0008
MARKER_AREA_FRAC_MAX: float = 0.002

# Legacy fixed protocols (kept so old metadata / GRPO_2 data still parse).
MARKER_REF_IMAGE_WH: tuple[int, int] = (2582, 1641)
# x45c: square whose diagonals *are* the X occupied ~0.5% of the page.
MARKER_AREA_FRAC_LEGACY: float = 0.005
# Relative to legacy area: large (x45d default) and small (GRPO_2 mix).
MARKER_AREA_SCALE_LARGE: float = 0.85
MARKER_AREA_SCALE_SMALL: float = 0.5
# Representative single-shot default = midpoint of the current range.
MARKER_AREA_FRAC: float = (MARKER_AREA_FRAC_MIN + MARKER_AREA_FRAC_MAX) / 2.0
MARKER_REF_SQUARE_AREA: float = MARKER_AREA_FRAC * (
    MARKER_REF_IMAGE_WH[0] * MARKER_REF_IMAGE_WH[1]
)
MARKER_REF_ARM_PX: int = int(round(math.sqrt(MARKER_REF_SQUARE_AREA / 2.0))) * 2
# Studio found translucent X unusable — protocol is fully opaque.
MARKER_ALPHA_AT_REF: int = 255
MARKER_ALPHA_MIN: int = 255


def sample_marker_area_frac(rng: Any) -> float:
    """Uniform random page-area fraction for one marked image (current protocol)."""
    lo, hi = float(MARKER_AREA_FRAC_MIN), float(MARKER_AREA_FRAC_MAX)
    if lo > hi:
        lo, hi = hi, lo
    return float(rng.uniform(lo, hi))


def area_frac_for_scale(area_scale: float) -> float:
    """Legacy: page-area fraction for a multiplier relative to x45c (0.005)."""
    return MARKER_AREA_FRAC_LEGACY * max(0.05, float(area_scale))


def marker_square_area(image_w: int, image_h: int, *, area_frac: float = MARKER_AREA_FRAC) -> float:
    """Area of the square whose diagonals are the X, in pixels^2."""
    return float(area_frac) * max(1, int(image_w)) * max(1, int(image_h))


def marker_opacity_for_area(square_area: float) -> int:
    """Always fully opaque (α=255). ``square_area`` kept for call-site compatibility."""
    del square_area
    return 255


def spec_for_image(
    image_w: int,
    image_h: int,
    *,
    base: MarkerSpec = MARKER_SPEC,
    area_frac: float = MARKER_AREA_FRAC,
    alpha: int | None = None,
    scale: float | None = None,
) -> MarkerSpec:
    """Page-relative X: constant area fraction, fully opaque unless ``alpha`` is set.

    ``scale`` if set is a multiplier on the protocol size (1.0 = protocol).
    """
    square_area = marker_square_area(image_w, image_h, area_frac=area_frac)
    if scale is not None:
        square_area *= max(0.05, float(scale)) ** 2
    half = max(6, int(round(math.sqrt(square_area / 2.0))))
    geom_scale = half / max(1, int(base.cross_half_length_px))
    a = 255 if alpha is None else max(0, min(255, int(alpha)))

    def _rgba(c: tuple[int, int, int, int], alpha: int) -> tuple[int, int, int, int]:
        return (int(c[0]), int(c[1]), int(c[2]), alpha)

    return replace(
        base,
        ring_radius_px=max(4, int(round(base.ring_radius_px * geom_scale))),
        ring_width_px=max(1, int(round(base.ring_width_px * geom_scale))),
        ring_color=_rgba(base.ring_color, a),
        cross_half_length_px=half,
        cross_width_px=max(1, int(round(base.cross_width_px * geom_scale))),
        cross_color=_rgba(base.cross_color, a),
        cross_outline_width_px=max(1, int(round(base.cross_outline_width_px * geom_scale))),
        cross_outline_color=_rgba(base.cross_outline_color, a),
        center_dot_radius_px=max(1, int(round(base.center_dot_radius_px * geom_scale))),
        center_dot_color=_rgba(base.center_dot_color, a),
        min_visible_extent_px=max(24, int(round(base.min_visible_extent_px * geom_scale))),
    )


def _paste_rgba(dst: Image.Image, src: Image.Image, dest_xy: tuple[int, int]) -> None:
    """Composite ``src`` onto ``dst``; clip when the patch hangs off the image edge."""
    dx, dy = dest_xy
    sw, sh = src.size
    dw, dh = dst.size
    sx0 = 0 if dx >= 0 else -dx
    sy0 = 0 if dy >= 0 else -dy
    dx0 = max(dx, 0)
    dy0 = max(dy, 0)
    sx1 = min(sw, dw - dx0 + sx0)
    sy1 = min(sh, dh - dy0 + sy0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    cropped = src.crop((sx0, sy0, sx1, sy1))
    dst.alpha_composite(cropped, (dx0, dy0))


def scale_dot_spec_for_image(
    image_w: int,
    image_h: int,
    base: DotMarkerSpec = MARKER_SPEC_V2,
) -> DotMarkerSpec:
    """Scale V2 marker with short side so it stays ViT-visible across window sizes."""
    ref = 1080.0
    short = float(min(max(1, image_w), max(1, image_h)))
    s = short / ref
    s = max(0.55, min(1.7, s))
    return DotMarkerSpec(
        radius_px=max(10, int(round(base.radius_px * s))),
        fill_alpha=base.fill_alpha,
        outline_width_px=max(2, int(round(base.outline_width_px * s))),
        outline_alpha=base.outline_alpha,
        sample_patch_px=max(20, int(round(base.sample_patch_px * s))),
        min_visible_extent_px=max(24, int(round(base.min_visible_extent_px * s))),
    )


def pick_marker_color(
    image: Image.Image,
    x: float,
    y: float,
    *,
    patch_px: int = 28,
) -> tuple[int, int, int]:
    """Pick an RGB fill color with high contrast vs local background near (x, y)."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    cx = int(round(x))
    cy = int(round(y))
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    r = max(4, int(patch_px))
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r + 1), min(h, cy + r + 1)
    patch = rgb.crop((x0, y0, x1, y1))
    pixels = list(patch.getdata())
    if not pixels:
        return (255, 40, 200)
    n = len(pixels)
    ar = sum(p[0] for p in pixels) / n
    ag = sum(p[1] for p in pixels) / n
    ab = sum(p[2] for p in pixels) / n
    lum = 0.2126 * ar + 0.7152 * ag + 0.0722 * ab
    if lum >= 140:
        return (20, 60, 220)
    if lum <= 80:
        return (255, 210, 40)
    cr, cg, cb = 255 - ar, 255 - ag, 255 - ab
    mx = max(cr, cg, cb)
    if mx < 1:
        return (255, 0, 180)
    scale = 255.0 / mx
    cr, cg, cb = cr * scale, cg * scale, cb * scale
    cr = min(255, int(0.55 * cr + 0.45 * 255))
    cg = min(255, int(0.55 * cg + 0.45 * 40))
    cb = min(255, int(0.55 * cb + 0.45 * 200))
    return (cr, cg, cb)


def draw_translucent_dot(
    image: Image.Image,
    x: float,
    y: float,
    spec: DotMarkerSpec = MARKER_SPEC_V2,
    *,
    copy: bool = True,
    color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Overlay a semi-opaque adaptive circle with solid outline (uniform fill, no center pin)."""
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

    rgb = color or pick_marker_color(base, cx, cy, patch_px=spec.sample_patch_px)
    fill = (rgb[0], rgb[1], rgb[2], int(spec.fill_alpha))
    # Solid outline: white on dark fills, near-black on light fills (max local contrast).
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    if lum >= 140:
        oc = (20, 20, 24)
    else:
        oc = (255, 255, 255)
    outline = (oc[0], oc[1], oc[2], int(spec.outline_alpha))

    r = int(spec.radius_px)
    box = [cx - r, cy - r, cx + r, cy + r]
    # Draw fill then a separate opaque stroke so outline_alpha=255 stays solid.
    draw.ellipse(box, fill=fill)
    ow = max(1, int(spec.outline_width_px))
    draw.ellipse(box, outline=outline, width=ow)

    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_crosshair(
    image: Image.Image,
    x: float,
    y: float,
    spec: MarkerSpec = MARKER_SPEC,
    *,
    copy: bool = True,
) -> Image.Image:
    """Overlay a magenta/white X at (x, y). Pass ``spec_for_image(w, h)`` in production."""
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

    r = spec.ring_radius_px
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=spec.ring_color,
        width=spec.ring_width_px,
    )

    pad = (
        spec.cross_half_length_px
        + spec.cross_width_px
        + 2 * spec.cross_outline_width_px
        + 4
    )
    side = pad * 2 + 1
    patch = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    pd = ImageDraw.Draw(patch)
    pc = side // 2
    half = spec.cross_half_length_px
    ow = spec.cross_width_px + 2 * spec.cross_outline_width_px
    pd.line([(pc - half, pc), (pc + half, pc)], fill=spec.cross_outline_color, width=ow)
    pd.line([(pc, pc - half), (pc, pc + half)], fill=spec.cross_outline_color, width=ow)
    pd.line([(pc - half, pc), (pc + half, pc)], fill=spec.cross_color, width=spec.cross_width_px)
    pd.line([(pc, pc - half), (pc, pc + half)], fill=spec.cross_color, width=spec.cross_width_px)
    rot = float(spec.rotation_deg)
    if abs(rot) > 0.01:
        patch = patch.rotate(rot, resample=Image.Resampling.BICUBIC, center=(pc, pc))
    _paste_rgba(overlay, patch, (cx - pc, cy - pc))

    d = spec.center_dot_radius_px
    ImageDraw.Draw(overlay).ellipse(
        [cx - d, cy - d, cx + d, cy + d], fill=spec.center_dot_color
    )

    return Image.alpha_composite(base, overlay).convert("RGB")


def _stamp_dot(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    fill: tuple[int, int, int, int],
    radius: int,
) -> None:
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill)


def draw_scatter_star(
    image: Image.Image,
    x: float,
    y: float,
    spec: ScatterStarSpec = MARKER_SPEC_A2,
    *,
    copy: bool = True,
) -> Image.Image:
    """Overlay a four-pointed star made of evenly spaced dots (A2 legacy)."""
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

    def put(px: float, py: float) -> None:
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < w and 0 <= iy < h):
            return
        if spec.outline_radius_px > 0:
            _stamp_dot(draw, ix, iy, fill=spec.outline_color, radius=spec.outline_radius_px)
        _stamp_dot(draw, ix, iy, fill=spec.dot_color, radius=spec.dot_radius_px)

    half = float(spec.arm_half_length_px)
    step = max(2.0, float(spec.dot_spacing_px))
    arms = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
    t = step
    while t <= half + 1e-6:
        for dx, dy in arms:
            put(cx + dx * t, cy + dy * t)
        t += step

    if spec.draw_center_dot:
        put(cx, cy)

    rr = float(spec.ring_radius_px)
    n = max(8, int(spec.ring_dot_count))
    ang0 = math.pi / n
    for i in range(n):
        ang = ang0 + 2.0 * math.pi * i / n
        put(cx + rr * math.cos(ang), cy + rr * math.sin(ang))

    return Image.alpha_composite(base, overlay).convert("RGB")


def marker_extent_ok(
    image_w: int,
    image_h: int,
    spec: MarkerSpec | ScatterStarSpec | DotMarkerSpec = MARKER_SPEC,
) -> bool:
    """True if the image is large enough for the marker to remain visible after mild resize."""
    if isinstance(spec, ScatterStarSpec):
        extent = max(spec.ring_radius_px * 2, spec.arm_half_length_px * 2)
        min_vis = spec.min_visible_extent_px
    elif isinstance(spec, DotMarkerSpec):
        extent = spec.radius_px * 2 + 4
        min_vis = spec.min_visible_extent_px
    else:
        adapted = spec_for_image(
            image_w, image_h, base=spec if isinstance(spec, MarkerSpec) else MARKER_SPEC
        )
        extent = max(adapted.ring_radius_px * 2, adapted.cross_half_length_px * 2)
        min_vis = adapted.min_visible_extent_px
    return (
        image_w >= min_vis
        and image_h >= min_vis
        and extent <= min(image_w, image_h)
    )

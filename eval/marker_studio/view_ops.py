"""Crop / edge-blur for marker studio chrome ablations.

Crop removes chrome pixels and remaps the marker. Edge-blur keeps geometry
but makes chrome text unreadable — so you can tell whether the model is
reading the chrome or just attending to the band.
"""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFilter

_MIN_REMAIN = 64


def parse_insets(
    obj: Any,
    width: int,
    height: int,
    *,
    min_remain: int = _MIN_REMAIN,
) -> tuple[int, int, int, int]:
    """Return (top, right, bottom, left) clamped so a usable interior remains."""
    src = obj if isinstance(obj, dict) else {}

    def _one(key: str, cap: int) -> int:
        try:
            v = int(round(float(src.get(key) or 0)))
        except (TypeError, ValueError):
            v = 0
        return max(0, min(v, max(0, cap)))

    top = _one("top", height - min_remain)
    bottom = _one("bottom", height - min_remain - top)
    left = _one("left", width - min_remain)
    right = _one("right", width - min_remain - left)
    if height - top - bottom < min_remain:
        bottom = max(0, height - top - min_remain)
    if width - left - right < min_remain:
        right = max(0, width - left - min_remain)
    return top, right, bottom, left


def crop_image(
    img: Image.Image,
    top: int,
    right: int,
    bottom: int,
    left: int,
) -> tuple[Image.Image, int, int]:
    """Crop by insets. Returns (cropped, origin_x, origin_y) in original pixels."""
    w, h = img.size
    top, right, bottom, left = parse_insets(
        {"top": top, "right": right, "bottom": bottom, "left": left},
        w,
        h,
    )
    box = (left, top, w - right, h - bottom)
    if box == (0, 0, w, h):
        return img, 0, 0
    return img.crop(box), left, top


def edge_blur(
    img: Image.Image,
    top: int,
    right: int,
    bottom: int,
    left: int,
    radius: float,
) -> Image.Image:
    """Blur a frame around the image; interior stays sharp with a short feather."""
    if radius <= 0:
        return img
    w, h = img.size
    top, right, bottom, left = parse_insets(
        {"top": top, "right": right, "bottom": bottom, "left": left},
        w,
        h,
        min_remain=1,
    )
    if top + right + bottom + left <= 0:
        return img
    blurred = img.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    inner = (left, top, w - right, h - bottom)
    if inner[2] <= inner[0] or inner[3] <= inner[1]:
        return blurred
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle(inner, fill=255)
    band = max(top, right, bottom, left, 1)
    feather = max(1.0, min(band / 3.0, float(radius)))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return Image.composite(img, blurred, mask)


def apply_studio_view(
    img: Image.Image,
    *,
    crop: Any = None,
    blur: Any = None,
) -> tuple[Image.Image, int, int]:
    """Apply crop then edge-blur. Blur insets are on the cropped image."""
    w, h = img.size
    ct, cr, cb, cl = parse_insets(crop, w, h)
    out, ox, oy = crop_image(img, ct, cr, cb, cl)
    bw, bh = out.size
    spec = blur if isinstance(blur, dict) else {}
    try:
        radius = float(spec.get("radius") or 0)
    except (TypeError, ValueError):
        radius = 0.0
    bt, br, bb, bl = parse_insets(spec, bw, bh, min_remain=1)
    if radius > 0 and (bt + br + bb + bl) > 0:
        out = edge_blur(out, bt, br, bb, bl, radius)
    return out, ox, oy


def remap_point(x: float, y: float, origin_x: int, origin_y: int) -> tuple[float, float]:
    return x - origin_x, y - origin_y

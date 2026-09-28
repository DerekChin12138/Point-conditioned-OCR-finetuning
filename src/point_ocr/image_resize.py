"""Image resize aligned with OvisOCR2 / Qwen2-VL style processor bounds."""

from __future__ import annotations

import math

from PIL import Image

# Official OvisOCR2 inference defaults (model card / vLLM example).
OVIS_MIN_PIXELS = 448 * 448
OVIS_MAX_PIXELS = 2880 * 2880
OVIS_PATCH_FACTOR = 28


def smart_resize_hw(
    height: int,
    width: int,
    *,
    factor: int = OVIS_PATCH_FACTOR,
    min_pixels: int = OVIS_MIN_PIXELS,
    max_pixels: int = OVIS_MAX_PIXELS,
) -> tuple[int, int]:
    """Return (h, w) divisible by factor, within [min_pixels, max_pixels] area."""
    if height < factor or width < factor:
        if height < width:
            width = max(factor, round(factor / max(height, 1) * width))
            height = factor
        else:
            height = max(factor, round(factor / max(width, 1) * height))
            width = factor

    if max(height, width) / max(min(height, width), 1) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / max(height * width, 1))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    return int(h_bar), int(w_bar)


def resize_for_ovis(
    image: Image.Image,
    *,
    min_pixels: int = OVIS_MIN_PIXELS,
    max_pixels: int = OVIS_MAX_PIXELS,
    factor: int = OVIS_PATCH_FACTOR,
) -> Image.Image:
    """Resize PIL image like OvisOCR2 processor (preserve aspect, clamp total pixels)."""
    w, h = image.size
    nh, nw = smart_resize_hw(
        h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    if (nh, nw) == (h, w):
        return image
    return image.resize((nw, nh), Image.Resampling.BICUBIC)

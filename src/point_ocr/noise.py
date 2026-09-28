"""Screen-domain noise: JPEG compression, mild scale, etc."""

from __future__ import annotations

import io
import random

from PIL import Image


def apply_screen_noise(
    image: Image.Image,
    *,
    rng: random.Random | None = None,
    jpeg_q_range: tuple[int, int] = (55, 92),
    scale_range: tuple[float, float] = (0.75, 1.0),
) -> Image.Image:
    rng = rng or random.Random()
    img = image.convert("RGB")

    # Mild downscale then ALWAYS restore original size.
    # POINT labels are in screenshot pixel coords — leaving the image smaller
    # would desync the crosshair from block bboxes (looks like a far miss).
    w0, h0 = img.size
    if rng.random() < 0.7:
        s = rng.uniform(*scale_range)
        nw, nh = max(64, int(w0 * s)), max(64, int(h0 * s))
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        img = img.resize((w0, h0), Image.Resampling.BILINEAR)

    # JPEG round-trip
    if rng.random() < 0.85:
        q = rng.randint(*jpeg_q_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img

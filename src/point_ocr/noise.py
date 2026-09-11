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

    # Mild downscale then back (simulates capture / DPI mismatch)
    if rng.random() < 0.7:
        s = rng.uniform(*scale_range)
        w, h = img.size
        nw, nh = max(64, int(w * s)), max(64, int(h * s))
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        if rng.random() < 0.5:
            img = img.resize((w, h), Image.Resampling.BILINEAR)

    # JPEG round-trip
    if rng.random() < 0.85:
        q = rng.randint(*jpeg_q_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img

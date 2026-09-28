from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "marker_studio"))

from view_ops import apply_studio_view, parse_insets, remap_point  # noqa: E402


def _solid(w: int, h: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def test_parse_insets_keeps_minimum_interior():
    top, right, bottom, left = parse_insets(
        {"top": 900, "bottom": 900, "left": 0, "right": 0},
        200,
        200,
        min_remain=64,
    )
    assert top + bottom <= 200 - 64
    assert left == 0 and right == 0


def test_crop_remaps_point():
    img = _solid(200, 100, (10, 20, 30))
    out, ox, oy = apply_studio_view(img, crop={"top": 20, "left": 10, "right": 0, "bottom": 0})
    assert out.size == (190, 80)
    assert (ox, oy) == (10, 20)
    x, y = remap_point(50, 40, ox, oy)
    assert (x, y) == (40, 20)


def test_edge_blur_changes_border_keeps_center():
    img = _solid(120, 120, (0, 255, 0))
    for y in range(120):
        for x in range(120):
            if x < 20 or x >= 100 or y < 20 or y >= 100:
                img.putpixel((x, y), (255, 0, 0) if (x + y) % 2 == 0 else (0, 0, 255))
    out, _, _ = apply_studio_view(
        img,
        blur={"top": 20, "right": 20, "bottom": 20, "left": 20, "radius": 8},
    )
    assert out.size == img.size
    assert out.getpixel((60, 60)) == (0, 255, 0)
    assert out.getpixel((2, 2)) not in {(255, 0, 0), (0, 0, 255)}

"""Tests for OvisOCR2-aligned smart resize."""

from __future__ import annotations

from PIL import Image

from point_ocr.image_resize import OVIS_MAX_PIXELS, resize_for_ovis, smart_resize_hw


def test_smart_resize_respects_max_pixels():
    h, w = smart_resize_hw(2160, 3840, max_pixels=OVIS_MAX_PIXELS)
    assert h % 28 == 0 and w % 28 == 0
    assert h * w <= OVIS_MAX_PIXELS


def test_resize_for_ovis_keeps_small_image():
    img = Image.new("RGB", (800, 600), (0, 0, 0))
    out = resize_for_ovis(img)
    # may upscale slightly to min_pixels / factor grid
    assert out.size[0] % 28 == 0
    assert out.size[1] % 28 == 0

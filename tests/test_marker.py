"""Unit tests for marker drawing."""

from __future__ import annotations

from PIL import Image

from point_ocr.marker import MARKER_SPEC, draw_crosshair, marker_extent_ok


def test_marker_spec_frozen_defaults():
    assert MARKER_SPEC.cross_color[:3] == (255, 0, 255)
    assert MARKER_SPEC.ring_color[:3] == (255, 255, 255)
    assert MARKER_SPEC.cross_half_length_px >= 24


def test_draw_crosshair_changes_pixels():
    img = Image.new("RGB", (400, 300), (30, 30, 30))
    out = draw_crosshair(img, 200, 150)
    assert out.size == img.size
    # Center should not stay dark gray
    px = out.getpixel((200, 150))
    assert px[0] > 200 and px[2] > 200  # magenta-ish


def test_extent_guard():
    assert marker_extent_ok(800, 600)
    assert not marker_extent_ok(20, 20)

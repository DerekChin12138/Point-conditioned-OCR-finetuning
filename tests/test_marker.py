"""Unit tests for marker drawing."""

from __future__ import annotations

from PIL import Image

from point_ocr.marker import MARKER_SPEC, draw_crosshair, marker_extent_ok


def test_marker_spec_batch2_open_center():
    assert MARKER_SPEC.cross_color[:3] == (255, 0, 255)
    assert MARKER_SPEC.ring_color[:3] == (255, 255, 255)
    assert MARKER_SPEC.cross_half_length_px >= 24
    assert MARKER_SPEC.center_gap_radius_px >= 5
    assert MARKER_SPEC.center_dot_radius_px == 0
    # Fixed size — not bbox-adaptive
    assert MARKER_SPEC.ring_radius_px == 22
    assert MARKER_SPEC.cross_half_length_px == 26


def test_draw_crosshair_open_center_preserves_center_pixel():
    bg = (30, 30, 30)
    img = Image.new("RGB", (400, 300), bg)
    out = draw_crosshair(img, 200, 150)
    assert out.size == img.size
    # Center of interest stays readable (background), not solid magenta.
    assert out.getpixel((200, 150)) == bg
    # Arm pixel (right of gap) should be magenta-ish
    arm_x = 200 + MARKER_SPEC.center_gap_radius_px + 4
    px = out.getpixel((arm_x, 150))
    assert px[0] > 200 and px[2] > 200


def test_extent_guard():
    assert marker_extent_ok(800, 600)
    assert not marker_extent_ok(20, 20)

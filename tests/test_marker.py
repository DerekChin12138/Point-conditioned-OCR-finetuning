"""Unit tests for marker drawing."""

from __future__ import annotations

import random

from PIL import Image

from point_ocr.marker import (
    MARKER_AREA_FRAC,
    MARKER_AREA_FRAC_LEGACY,
    MARKER_AREA_FRAC_MAX,
    MARKER_AREA_FRAC_MIN,
    MARKER_AREA_SCALE_LARGE,
    MARKER_AREA_SCALE_SMALL,
    MARKER_ALPHA_AT_REF,
    MARKER_REF_ARM_PX,
    MARKER_REF_IMAGE_WH,
    MARKER_SPEC,
    MARKER_SPEC_A2,
    MARKER_SPEC_V2,
    CURRENT_MARKER_TAG,
    area_frac_for_scale,
    draw_crosshair,
    draw_scatter_star,
    draw_translucent_dot,
    marker_extent_ok,
    marker_opacity_for_area,
    marker_square_area,
    sample_marker_area_frac,
    spec_for_image,
)


def test_marker_spec_thick_crosshair():
    assert MARKER_SPEC.cross_color[:3] == (255, 0, 255)
    assert MARKER_SPEC.cross_color[3] == 255
    assert MARKER_SPEC.ring_color[:3] == (255, 255, 255)
    assert MARKER_SPEC.cross_half_length_px >= 24
    assert MARKER_SPEC.cross_width_px == 3
    assert MARKER_SPEC.ring_width_px == 3
    assert MARKER_SPEC.cross_outline_width_px == 1
    assert MARKER_SPEC.center_dot_radius_px >= 3
    assert MARKER_SPEC.ring_radius_px >= 24
    assert abs(MARKER_SPEC.rotation_deg - 45.0) < 1e-6


def test_v2_dot_is_vit_visible():
    """Failed A1_v2 run used ~r12/α88; require a stronger cue."""
    assert MARKER_SPEC_V2.radius_px >= 18
    assert MARKER_SPEC_V2.fill_alpha >= 180
    assert MARKER_SPEC_V2.outline_alpha >= 220
    assert MARKER_SPEC_V2.outline_width_px >= 2
    assert not hasattr(MARKER_SPEC_V2, "center_dot_radius_px") or getattr(
        MARKER_SPEC_V2, "center_dot_radius_px", 0
    ) in (0, None)
    img = Image.new("RGB", (400, 300), (240, 240, 240))
    out = draw_translucent_dot(img, 200, 150)
    px = out.getpixel((200, 150))
    # Strong blue adaptive fill on light bg (uniform disk, no darker pin)
    assert px[2] > 150 and px[0] < 120
    # Outline / fill should differ from plain background
    bg = out.getpixel((50, 50))
    assert abs(px[0] - bg[0]) + abs(px[1] - bg[1]) + abs(px[2] - bg[2]) > 120
    # Center should match near-center fill (no opaque pin)
    near = out.getpixel((200 + 8, 150))
    assert abs(px[0] - near[0]) + abs(px[1] - near[1]) + abs(px[2] - near[2]) < 40


def _is_magenta(px: tuple[int, int, int]) -> bool:
    return px[0] > 180 and px[2] > 180 and px[1] < 90


def test_draw_crosshair_changes_pixels():
    img = Image.new("RGB", (400, 300), (30, 30, 30))
    out = draw_crosshair(img, 200, 150)
    assert out.size == img.size
    # Center should not stay dark gray (solid magenta dot)
    px = out.getpixel((200, 150))
    assert px[0] > 200 and px[2] > 200


def test_draw_crosshair_is_rotated_x():
    img = Image.new("RGB", (400, 300), (30, 30, 30))
    out = draw_crosshair(img, 200, 150)
    # 45° arms: magenta along the diagonal, not along the old + axes
    found_diag = any(_is_magenta(out.getpixel((200 + t, 150 + t))) for t in range(8, 24))
    assert found_diag
    axis = out.getpixel((200 + 16, 150))
    assert not _is_magenta(axis)


def test_scatter_star_uses_round_sparse_dots():
    assert MARKER_SPEC_A2.dot_spacing_px >= 9.0
    assert MARKER_SPEC_A2.dot_radius_px >= 1
    assert MARKER_SPEC_A2.ring_dot_count <= 10
    assert MARKER_SPEC_A2.dot_color[3] < 255  # translucent
    img = Image.new("RGB", (400, 300), (30, 30, 30))
    out = draw_scatter_star(img, 200, 150)
    assert out.size == img.size
    px = out.getpixel((200, 150))
    # Semi-transparent magenta over dark bg → blended, still clearly tinted
    assert px[0] > 120 and px[2] > 120 and px[0] > px[1]
    # Mid-arm should be mostly background (gap between sparse dots)
    mid = out.getpixel((200 + 5, 150))
    assert mid[0] < 80 or mid[1] > 100  # not a solid magenta bar


def test_area_frac_is_random_range():
    """Current protocol: per-image area fraction is uniform in [MIN, MAX]."""
    assert abs(MARKER_AREA_FRAC_LEGACY - 0.005) < 1e-12
    assert abs(MARKER_AREA_SCALE_LARGE - 0.85) < 1e-12
    assert abs(MARKER_AREA_SCALE_SMALL - 0.5) < 1e-12
    assert MARKER_AREA_FRAC_MIN == 0.0008 and MARKER_AREA_FRAC_MAX == 0.002
    assert abs(MARKER_AREA_FRAC - (MARKER_AREA_FRAC_MIN + MARKER_AREA_FRAC_MAX) / 2) < 1e-12
    # Legacy scale helper is unchanged (old metadata still parses).
    assert abs(area_frac_for_scale(0.5) - MARKER_AREA_FRAC_LEGACY * 0.5) < 1e-12
    # Sampling stays inside the range and actually varies.
    rng = random.Random(0)
    vals = [sample_marker_area_frac(rng) for _ in range(200)]
    assert all(MARKER_AREA_FRAC_MIN <= v <= MARKER_AREA_FRAC_MAX for v in vals)
    assert max(vals) - min(vals) > (MARKER_AREA_FRAC_MAX - MARKER_AREA_FRAC_MIN) * 0.8
    # spec_for_image default draws the representative midpoint.
    w, h = MARKER_REF_IMAGE_WH
    spec = spec_for_image(w, h)
    assert spec.cross_half_length_px == MARKER_REF_ARM_PX // 2
    share = 2 * spec.cross_half_length_px**2 / (w * h)
    assert abs(share - MARKER_AREA_FRAC) < 0.0005
    spec_s = spec_for_image(1280, 720)
    share_s = 2 * spec_s.cross_half_length_px**2 / (1280 * 720)
    assert abs(share_s - MARKER_AREA_FRAC) < 0.003
    # Area ratios vs legacy arm (linear size scales with sqrt(area)).
    leg = spec_for_image(w, h, area_frac=MARKER_AREA_FRAC_LEGACY)
    small = spec_for_image(w, h, area_frac=area_frac_for_scale(MARKER_AREA_SCALE_SMALL))
    assert abs(small.cross_half_length_px / leg.cross_half_length_px - MARKER_AREA_SCALE_SMALL**0.5) < 0.02


def test_opacity_is_fully_opaque():
    assert marker_opacity_for_area(100.0) == 255
    assert marker_opacity_for_area(1e7) == 255
    assert spec_for_image(800, 600).cross_color[3] == 255
    assert spec_for_image(*MARKER_REF_IMAGE_WH).cross_color[3] == 255
    assert spec_for_image(*MARKER_REF_IMAGE_WH, alpha=120).cross_color[3] == 120


def test_current_tag_is_area_protocol():
    assert CURRENT_MARKER_TAG == "x45r"
    assert MARKER_ALPHA_AT_REF == 255


def test_extent_guard():
    assert marker_extent_ok(800, 600)
    assert not marker_extent_ok(20, 20)
    assert marker_extent_ok(800, 600, MARKER_SPEC_A2)
    assert marker_extent_ok(800, 600, MARKER_SPEC_V2)

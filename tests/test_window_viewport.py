"""Tests for focused-window viewport sampling."""

from __future__ import annotations

import random

from point_ocr.synth.window_viewport import (
    _MAX_ASPECT,
    _MAX_OUTPUT_PIXELS,
    _MIN_ASPECT,
    sample_window_capture,
)


def test_sample_window_capture_diverse():
    rng = random.Random(0)
    specs = [sample_window_capture(rng) for _ in range(400)]
    widths = {s.width for s in specs}
    heights = {s.height for s in specs}
    aspects = [s.aspect for s in specs]
    dprs = {s.device_scale_factor for s in specs}
    assert len(widths) >= 25
    assert len(heights) >= 25
    assert min(aspects) < 0.95  # some portrait / square
    assert max(aspects) > 1.4  # some landscape
    assert 1.0 in dprs
    assert any(d > 1.0 for d in dprs)
    assert any(not s.full_page for s in specs)
    assert all(not s.full_page for s in specs)
    assert all(s.output_pixels <= _MAX_OUTPUT_PIXELS for s in specs)
    assert all(s.width >= 400 and s.height >= 400 for s in specs)


def test_sample_window_capture_aspect_stays_natural():
    rng = random.Random(7)
    specs = [sample_window_capture(rng, full_page_prob=0.2) for _ in range(500)]
    slack = 0.04  # snap-to-8 px
    lo, hi = _MIN_ASPECT - slack, _MAX_ASPECT + slack
    for s in specs:
        assert lo <= s.aspect <= hi, (s.viewport, s.aspect, s.family)


def test_sample_window_capture_min_floors():
    rng = random.Random(3)
    specs = [
        sample_window_capture(rng, min_css_pixels=1024 * 720, min_aspect=1.0) for _ in range(80)
    ]
    assert all(s.css_pixels >= 1024 * 720 - 64 for s in specs)
    assert all(s.aspect >= 0.99 for s in specs)


def test_sample_window_capture_meta_keys():
    s = sample_window_capture(random.Random(1))
    meta = s.to_meta()
    assert meta["viewport"] == [s.width, s.height]
    assert "aspect" in meta and "device_scale_factor" in meta
    assert "full_page" in meta

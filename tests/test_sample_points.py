"""Point sampling tests."""

from __future__ import annotations

import random

from point_ocr.sample_points import (
    BBox,
    points_per_block_range,
    sample_negative_points,
    sample_points_in_block,
)


def test_sample_inside_box():
    box = BBox(100, 100, 300, 220)
    rng = random.Random(0)
    pts = sample_points_in_block(box, n=5, rng=rng, block_id="b1")
    assert len(pts) == 5
    for p in pts:
        assert box.contains(p.x, p.y)
        assert p.kind == "positive"
        assert p.block_id == "b1"


def test_corners_spread_on_wide_box():
    """Wide short line: points should not all collapse to horizontal center."""
    box = BBox(0, 0, 400, 40)
    rng = random.Random(7)
    pts = sample_points_in_block(box, n=4, rng=rng, coverage="corners")
    xs = [p.x for p in pts]
    assert max(xs) - min(xs) > box.width() * 0.35
    regions = {p.region for p in pts}
    assert len(regions) >= 2


def test_negatives_outside():
    boxes = [BBox(50, 50, 200, 200), BBox(250, 80, 400, 300)]
    rng = random.Random(1)
    negs = sample_negative_points(500, 400, boxes, n=6, rng=rng)
    assert len(negs) >= 1
    for p in negs:
        assert p.kind == "negative"
        assert not any(b.contains(p.x, p.y) for b in boxes)


def test_r_range():
    rng = random.Random(2)
    vals = {points_per_block_range(2, 5, rng=rng) for _ in range(40)}
    assert vals <= {2, 3, 4, 5}
    assert len(vals) >= 2

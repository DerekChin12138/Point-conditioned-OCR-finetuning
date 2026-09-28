"""Point sampling tests."""

from __future__ import annotations

import random

from point_ocr.sample_points import (
    BBox,
    DEFAULT_A1_BAND_HALF_FRAC,
    DEFAULT_IMAGE_EDGE_MARGIN_PX,
    center_band_box,
    diamond5_region,
    inner_area_box,
    near_image_edge,
    points_per_block_range,
    sample_diamond5_points,
    sample_negative_points,
    sample_points_in_block,
    sample_points_in_fragments,
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
        assert p.region == "center_band"


def test_center_band_within_035():
    box = BBox(0, 0, 400, 200)
    band = center_band_box(box, band_half_frac=0.35)
    assert abs(band.x0 - 60.0) < 1e-6  # 0.15 * 400
    assert abs(band.x1 - 340.0) < 1e-6
    assert abs(band.y0 - 30.0) < 1e-6
    assert abs(band.y1 - 170.0) < 1e-6
    rng = random.Random(7)
    pts = sample_points_in_block(box, n=20, rng=rng, coverage="center_band")
    assert len(pts) == 20
    for p in pts:
        assert band.x0 - 1e-6 <= p.x <= band.x1 + 1e-6
        assert band.y0 - 1e-6 <= p.y <= band.y1 + 1e-6
        # Not glued to exact center
    xs = [p.x for p in pts]
    assert max(xs) - min(xs) > box.width() * 0.15


def test_center_heavy_legacy_still_works():
    box = BBox(0, 0, 400, 200)
    rng = random.Random(7)
    pts = sample_points_in_block(box, n=8, rng=rng, coverage="center_heavy")
    assert len(pts) == 8
    for p in pts:
        assert 0.2 * box.width() < (p.x - box.x0) < 0.8 * box.width()
        assert 0.2 * box.height() < (p.y - box.y0) < 0.8 * box.height()


def test_fragments_never_sample_gutter():
    """Two column fragments with a gutter in between — points must land on ink."""
    left = BBox(0, 200, 100, 280)
    right = BBox(140, 40, 240, 120)
    gutter_x = 120  # between columns
    rng = random.Random(3)
    pts = sample_points_in_fragments([left, right], n=16, rng=rng, coverage="center_band")
    assert len(pts) == 16
    for p in pts:
        assert left.contains(p.x, p.y) or right.contains(p.x, p.y)
        assert abs(p.x - gutter_x) > 15
        assert p.fragment_index in {0, 1}
        assert p.fragment_bbox is not None


def test_corners_spread_on_wide_box():
    """Wide short line: corners mode still spreads horizontally."""
    box = BBox(0, 0, 400, 40)
    rng = random.Random(7)
    pts = sample_points_in_block(box, n=4, rng=rng, coverage="corners")
    xs = [p.x for p in pts]
    assert max(xs) - min(xs) > box.width() * 0.35
    regions = {p.region for p in pts}
    assert len(regions) >= 2


def test_corner_extreme_near_box_corners():
    """Extreme mode lands near ink-box corners, not the center band."""
    box = BBox(0, 0, 400, 300)
    rng = random.Random(11)
    pts = sample_points_in_block(box, n=8, rng=rng, coverage="corner_extreme")
    assert len(pts) == 8
    band = center_band_box(box, band_half_frac=0.35)
    outside_band = 0
    for p in pts:
        assert box.contains(p.x, p.y)
        assert p.region and p.region.startswith("ext_")
        if not (band.x0 <= p.x <= band.x1 and band.y0 <= p.y <= band.y1):
            outside_band += 1
    assert outside_band >= 5


def test_edge_band_stays_near_border():
    box = BBox(0, 0, 400, 300)
    rng = random.Random(3)
    pts = sample_points_in_block(box, n=20, rng=rng, coverage="edge_band")
    assert len(pts) == 20
    for p in pts:
        assert box.contains(p.x, p.y)
        assert p.region and p.region.startswith("edge_")
        # At least one coordinate near a border (within 25% of that side)
        nx = (p.x - box.x0) / box.width()
        ny = (p.y - box.y0) / box.height()
        near = nx <= 0.25 or nx >= 0.75 or ny <= 0.25 or ny >= 0.75
        assert near


def test_positive_rejects_image_edge():
    box = BBox(0, 0, 40, 40)  # entirely inside the default edge margin band
    rng = random.Random(0)
    pts = sample_points_in_block(
        box,
        n=5,
        rng=rng,
        coverage="center_band",
        image_w=800,
        image_h=600,
        image_edge_margin=48,
    )
    assert len(pts) == 0


def test_negatives_away_from_edge_and_text():
    boxes = [BBox(50, 50, 200, 200), BBox(250, 80, 400, 300)]
    rng = random.Random(1)
    negs = sample_negative_points(500, 400, boxes, n=6, margin=48, clearance=24, rng=rng)
    assert len(negs) >= 1
    for p in negs:
        assert p.kind == "negative"
        assert not near_image_edge(p.x, p.y, 500, 400, margin=47)
        assert not any(b.contains(p.x, p.y) for b in boxes)


def test_r_range():
    rng = random.Random(2)
    vals = {points_per_block_range(2, 5, rng=rng) for _ in range(40)}
    assert vals <= {2, 3, 4, 5}
    assert len(vals) >= 2


def test_near_image_edge_helper():
    assert near_image_edge(10, 100, 800, 600, margin=48)
    assert not near_image_edge(100, 100, 800, 600, margin=48)
    assert DEFAULT_IMAGE_EDGE_MARGIN_PX >= 40
    assert DEFAULT_A1_BAND_HALF_FRAC == 0.35


def test_inner_area_box_is_80pct():
    from point_ocr.sample_points import inner_area_box, inner_area_inset_frac

    box = BBox(0, 0, 400, 200)
    inner = inner_area_box(box, area_frac=0.80)
    assert abs(inner.area() / box.area() - 0.80) < 0.02
    m = inner_area_inset_frac(0.80)
    assert abs(m - 0.052786) < 1e-4


def test_inner_area_samples_stay_inside():
    from point_ocr.sample_points import inner_area_box

    box = BBox(80, 80, 480, 320)
    inner = inner_area_box(box)
    rng = random.Random(4)
    pts = sample_points_in_block(box, n=30, rng=rng, coverage="inner_area")
    assert len(pts) == 30
    for p in pts:
        assert inner.contains(p.x, p.y)
        assert p.region == "inner_area"


def test_diamond5_covers_at_least_three_regions():
    box = BBox(0, 0, 400, 300)
    inner = inner_area_box(box)
    rng = random.Random(0)
    pts = sample_diamond5_points(box, n_regions=3, points_per_region=2, rng=rng)
    assert len(pts) == 6
    regions = {p.region for p in pts}
    assert "diamond" in regions
    assert len(regions) == 3
    counts = {r: 0 for r in regions}
    for p in pts:
        assert inner.contains(p.x, p.y)
        assert diamond5_region(inner, p.x, p.y) == p.region
        counts[p.region] += 1
    assert all(c == 2 for c in counts.values())


def test_diamond5_uses_all_five_regions():
    box = BBox(0, 0, 400, 300)
    inner = inner_area_box(box)
    pts = sample_diamond5_points(box, rng=random.Random(0))
    assert len(pts) == 10
    counts: dict[str, int] = {}
    for p in pts:
        assert inner.contains(p.x, p.y)
        assert diamond5_region(inner, p.x, p.y) == p.region
        counts[p.region] = counts.get(p.region, 0) + 1
    assert set(counts) == {"diamond", "nw", "ne", "sw", "se"}
    assert all(c == 2 for c in counts.values())


def test_diamond5_regions_partition_inner_box():
    box = BBox(40, 20, 360, 280)
    inner = inner_area_box(box)
    seen = set()
    rng = random.Random(1)
    for _ in range(400):
        x = rng.uniform(inner.x0 + 0.5, inner.x1 - 0.5)
        y = rng.uniform(inner.y0 + 0.5, inner.y1 - 0.5)
        seen.add(diamond5_region(inner, x, y))
    assert seen == {"diamond", "nw", "ne", "sw", "se"}
    # Side midpoints of the inner box sit on the diamond boundary.
    cx = (inner.x0 + inner.x1) / 2
    cy = (inner.y0 + inner.y1) / 2
    assert diamond5_region(inner, cx, inner.y0 + 0.1) == "diamond"
    assert diamond5_region(inner, inner.x0 + 1, inner.y0 + 1) == "nw"


def test_diamond5_via_coverage_flag():
    box = BBox(0, 0, 500, 360)
    pts = sample_points_in_block(box, n=6, rng=random.Random(2), coverage="diamond5")
    assert len(pts) == 10
    assert {p.region for p in pts} == {"diamond", "nw", "ne", "sw", "se"}


def test_boundary_empty_outside_full_bbox():
    from point_ocr.sample_points import sample_boundary_empty_points

    left = BBox(80, 80, 280, 400)
    right = BBox(300, 80, 500, 400)  # 20px gutter
    rng = random.Random(0)
    pts = sample_boundary_empty_points(800, 600, [left, right], n=12, rng=rng)
    assert len(pts) >= 4
    for p in pts:
        assert p.kind == "negative"
        assert p.region == "neg_boundary"
        assert not left.contains(p.x, p.y)
        assert not right.contains(p.x, p.y)


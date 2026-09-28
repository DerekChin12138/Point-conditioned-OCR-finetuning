"""Sample interest points inside block bboxes + negative (empty-label) points.

Product hit rule: marker center inside the inner 80% *area* of a text bbox
(``coverage="inner_area"``). Clear off-text and two-block junctions that miss
every inner area are empty-label negatives.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates: [x0, y0, x1, y1), exclusive end."""

    x0: float
    y0: float
    x1: float
    y1: float

    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    def area(self) -> float:
        return self.width() * self.height()

    def contains(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        return (
            self.x0 + margin <= x < self.x1 - margin
            and self.y0 + margin <= y < self.y1 - margin
        )

    def expand(self, pad: float) -> BBox:
        return BBox(self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad)

    def clamp_point(self, x: float, y: float, *, inset: float = 1.0) -> tuple[float, float]:
        x = min(max(x, self.x0 + inset), max(self.x0 + inset, self.x1 - inset))
        y = min(max(y, self.y0 + inset), max(self.y0 + inset, self.y1 - inset))
        return x, y

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


def union_bbox(boxes: Sequence[BBox]) -> BBox | None:
    """Axis-aligned union, or None if ``boxes`` is empty."""
    acc: list[BBox] = [b for b in boxes if b.width() > 0 and b.height() > 0]
    if not acc:
        return None
    return BBox(
        min(b.x0 for b in acc),
        min(b.y0 for b in acc),
        max(b.x1 for b in acc),
        max(b.y1 for b in acc),
    )


@dataclass(frozen=True)
class SampledPoint:
    x: float
    y: float
    kind: str  # "positive" | "negative"
    block_id: str | None = None
    region: str | None = None  # e.g. "center_band", "mid_left"
    fragment_index: int | None = None
    fragment_bbox: BBox | None = None


# Corner / edge anchors — only for coverage="corners" (A2+ stress; not A1 default).
_CORNER_ANCHORS: list[tuple[str, float, float]] = [
    ("nw", 0.12, 0.18),
    ("ne", 0.88, 0.18),
    ("sw", 0.12, 0.82),
    ("se", 0.88, 0.82),
    ("west", 0.10, 0.50),
    ("east", 0.90, 0.50),
    ("north", 0.50, 0.15),
    ("south", 0.50, 0.85),
    ("center", 0.50, 0.50),
    ("mid_left", 0.30, 0.50),
    ("mid_right", 0.70, 0.50),
]

# Extreme corners of the *ink* box (A2 dedicated slice). Rel u,v in [0,1] of near-full box.
_CORNER_EXTREME_ANCHORS: list[tuple[str, float, float]] = [
    ("ext_nw", 0.06, 0.08),
    ("ext_ne", 0.94, 0.08),
    ("ext_sw", 0.06, 0.92),
    ("ext_se", 0.94, 0.92),
    ("ext_west", 0.05, 0.50),
    ("ext_east", 0.95, 0.50),
    ("ext_north", 0.50, 0.06),
    ("ext_south", 0.50, 0.94),
]

# Legacy tight center anchors (kept for coverage="center_heavy").
_CENTER_HEAVY_ANCHORS: list[tuple[str, float, float]] = [
    ("center", 0.50, 0.50),
    ("center", 0.50, 0.50),
    ("center", 0.50, 0.50),
    ("mid_left", 0.42, 0.50),
    ("mid_right", 0.58, 0.50),
    ("mid_up", 0.50, 0.42),
    ("mid_down", 0.50, 0.58),
]

# Keep alias for older imports/tests.
_REGION_ANCHORS = _CORNER_ANCHORS

# Defaults tuned after dense-preview human review.
DEFAULT_IMAGE_EDGE_MARGIN_PX = 48.0
DEFAULT_NEG_CLEARANCE_PX = 28.0
# Q1 SFT empties: obvious AABB gap only (Chebyshev / expand). Junctions stay out.
Q1_NEG_CLEARANCE_PX = 48.0
# A1: sample in center ± band_half_frac * box size (so inset = 0.5 - band).
DEFAULT_A1_BAND_HALF_FRAC = 0.35
# Product rule: a hit is the marker center inside the inner 80% *area* of a bbox.
# Uniform inset: (1 - 2m)^2 = 0.80 → m = (1 - √0.80) / 2 ≈ 0.0528.
DEFAULT_INNER_AREA_FRAC = 0.80
DEFAULT_BOUNDARY_MAX_GAP_PX = 48.0


def _inner_box(box: BBox, inset_frac_x: float, inset_frac_y: float) -> BBox:
    inset_x = max(1.0, box.width() * inset_frac_x)
    inset_y = max(1.0, box.height() * inset_frac_y)
    if box.width() - 2 * inset_x < 2:
        inset_x = max(0.5, box.width() * 0.04)
    if box.height() - 2 * inset_y < 2:
        inset_y = max(0.5, box.height() * 0.08)
    return BBox(
        box.x0 + inset_x,
        box.y0 + inset_y,
        box.x1 - inset_x,
        box.y1 - inset_y,
    )


def center_band_box(box: BBox, *, band_half_frac: float = DEFAULT_A1_BAND_HALF_FRAC) -> BBox:
    """Inner box = center ± band_half_frac * (width|height)."""
    half = max(0.05, min(0.49, float(band_half_frac)))
    inset = 0.5 - half
    return _inner_box(box, inset, inset)


def inner_area_inset_frac(area_frac: float = DEFAULT_INNER_AREA_FRAC) -> float:
    """Uniform inset so the remaining rectangle has ``area_frac`` of the bbox area."""
    af = max(0.05, min(0.98, float(area_frac)))
    return (1.0 - math.sqrt(af)) / 2.0


def inner_area_box(box: BBox, *, area_frac: float = DEFAULT_INNER_AREA_FRAC) -> BBox:
    """Inner rectangle covering ``area_frac`` of ``box`` (default 80%), centered."""
    m = inner_area_inset_frac(area_frac)
    return _inner_box(box, m, m)


DIAMOND5_REGIONS: tuple[str, ...] = ("diamond", "nw", "ne", "sw", "se")
DEFAULT_DIAMOND5_N_REGIONS = 5
DEFAULT_DIAMOND5_POINTS_PER_REGION = 2


def diamond5_region(box: BBox, x: float, y: float) -> str:
    """Which of the five diamond-split regions contains (x, y) inside ``box``.

    Midpoints of the four sides are joined into a diamond (Manhattan ellipse).
    """
    cx = (box.x0 + box.x1) / 2.0
    cy = (box.y0 + box.y1) / 2.0
    hw = max(box.width() / 2.0, 1e-6)
    hh = max(box.height() / 2.0, 1e-6)
    if abs(x - cx) / hw + abs(y - cy) / hh <= 1.0:
        return "diamond"
    if x < cx and y < cy:
        return "nw"
    if x >= cx and y < cy:
        return "ne"
    if x < cx:
        return "sw"
    return "se"


def _sample_in_diamond5_region(
    box: BBox,
    region: str,
    rng: random.Random,
    *,
    max_tries: int = 80,
) -> tuple[float, float] | None:
    if box.width() < 2 or box.height() < 2:
        return None
    for _ in range(max_tries):
        x = rng.uniform(box.x0, box.x1)
        y = rng.uniform(box.y0, box.y1)
        if diamond5_region(box, x, y) == region:
            return x, y
    # Degenerate fallback: region centroid-ish.
    cx = (box.x0 + box.x1) / 2.0
    cy = (box.y0 + box.y1) / 2.0
    qx = (box.x0 + cx) / 2.0 if region in {"nw", "sw"} else (cx + box.x1) / 2.0
    qy = (box.y0 + cy) / 2.0 if region in {"nw", "ne"} else (cy + box.y1) / 2.0
    if region == "diamond":
        qx, qy = cx, cy
    if diamond5_region(box, qx, qy) != region:
        return None
    return qx, qy


def sample_diamond5_points(
    box: BBox,
    *,
    n_regions: int = DEFAULT_DIAMOND5_N_REGIONS,
    points_per_region: int = DEFAULT_DIAMOND5_POINTS_PER_REGION,
    rng: random.Random | None = None,
    block_id: str | None = None,
    image_w: int | None = None,
    image_h: int | None = None,
    image_edge_margin: float = DEFAULT_IMAGE_EDGE_MARGIN_PX,
    fragment_index: int | None = None,
) -> list[SampledPoint]:
    """All 5 inner-80% regions × 2 uniform points (product-hit safe).

    ``n_regions < 5`` keeps the old subsample (diamond + random corners).
    """
    rng = rng or random.Random()
    inner = inner_area_box(box)
    if inner.width() < 4 or inner.height() < 4:
        return []
    k = max(1, min(5, int(n_regions)))
    n_pp = max(1, int(points_per_region))
    if k >= 5:
        chosen = list(DIAMOND5_REGIONS)
        rng.shuffle(chosen)
    else:
        corners = ["nw", "ne", "sw", "se"]
        rng.shuffle(corners)
        chosen = ["diamond"] + corners[: max(0, k - 1)]
        rng.shuffle(chosen)

    def _accept(x: float, y: float) -> bool:
        if image_w is None or image_h is None:
            return True
        return not near_image_edge(x, y, image_w, image_h, margin=image_edge_margin)

    out: list[SampledPoint] = []
    for region in chosen:
        got = 0
        tries = 0
        while got < n_pp and tries < n_pp * 60:
            tries += 1
            xy = _sample_in_diamond5_region(inner, region, rng)
            if xy is None:
                break
            x, y = inner.clamp_point(xy[0], xy[1], inset=0.5)
            if diamond5_region(inner, x, y) != region:
                continue
            if not _accept(x, y):
                continue
            out.append(
                SampledPoint(
                    x=x,
                    y=y,
                    kind="positive",
                    block_id=block_id,
                    region=region,
                    fragment_index=fragment_index,
                    fragment_bbox=box,
                )
            )
            got += 1
    return out


def point_in_inner_area(
    x: float,
    y: float,
    box: BBox,
    *,
    area_frac: float = DEFAULT_INNER_AREA_FRAC,
) -> bool:
    return inner_area_box(box, area_frac=area_frac).contains(x, y)


def facing_gap_px(a: BBox, b: BBox) -> float | None:
    """Gap between two axis-aligned boxes that share a facing edge.

    Returns 0 if they overlap on the facing axis; ``None`` if they are not
    stacked or side-by-side (diagonal neighbors).
    """
    x_overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    y_overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if x_overlap > 4:
        if a.y1 <= b.y0:
            return float(b.y0 - a.y1)
        if b.y1 <= a.y0:
            return float(a.y0 - b.y1)
        return 0.0
    if y_overlap > 4:
        if a.x1 <= b.x0:
            return float(b.x0 - a.x1)
        if b.x1 <= a.x0:
            return float(a.x0 - b.x1)
        return 0.0
    return None


def gap_strip(a: BBox, b: BBox) -> BBox | None:
    """The rectangle between two facing boxes (empty if they touch/overlap)."""
    x_overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    y_overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if x_overlap > 4:
        x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
        if a.y1 < b.y0:
            return BBox(x0, a.y1, x1, b.y0)
        if b.y1 < a.y0:
            return BBox(x0, b.y1, x1, a.y0)
        return None
    if y_overlap > 4:
        y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
        if a.x1 < b.x0:
            return BBox(a.x1, y0, b.x0, y1)
        if b.x1 < a.x0:
            return BBox(b.x1, y0, a.x0, y1)
        return None
    return None


def _sample_edge_band_point(
    box: BBox,
    rng: random.Random,
    *,
    inner_frac: float = 0.08,
    outer_frac: float = 0.22,
) -> tuple[float, float, str] | None:
    """Sample one point in the ink-box frame between inner_frac and outer_frac from the border."""
    lo = max(0.02, min(0.45, float(inner_frac)))
    hi = max(lo + 0.02, min(0.49, float(outer_frac)))
    w, h = box.width(), box.height()
    if w < 8 or h < 8:
        return None
    side = rng.choice(["north", "south", "west", "east"])
    t = rng.uniform(lo, hi)
    u = rng.uniform(0.15, 0.85)
    if side == "north":
        x = box.x0 + u * w
        y = box.y0 + t * h
    elif side == "south":
        x = box.x0 + u * w
        y = box.y1 - t * h
    elif side == "west":
        x = box.x0 + t * w
        y = box.y0 + u * h
    else:
        x = box.x1 - t * w
        y = box.y0 + u * h
    x, y = box.clamp_point(x, y, inset=0.5)
    return x, y, f"edge_{side}"


def near_image_edge(
    x: float,
    y: float,
    image_w: int,
    image_h: int,
    *,
    margin: float = DEFAULT_IMAGE_EDGE_MARGIN_PX,
) -> bool:
    return (
        x < margin
        or y < margin
        or x >= image_w - margin
        or y >= image_h - margin
    )


def _min_dist_outside_boxes(x: float, y: float, boxes: Sequence[BBox]) -> float:
    """0 if inside any box; else Euclidean distance to nearest box edge."""
    best = float("inf")
    for b in boxes:
        if b.contains(x, y):
            return 0.0
        dx = 0.0
        if x < b.x0:
            dx = b.x0 - x
        elif x >= b.x1:
            dx = x - (b.x1 - 1e-6)
        dy = 0.0
        if y < b.y0:
            dy = b.y0 - y
        elif y >= b.y1:
            dy = y - (b.y1 - 1e-6)
        best = min(best, math.hypot(dx, dy))
    return best if best < float("inf") else float("inf")


def sample_points_in_block(
    box: BBox,
    *,
    n: int = 3,
    center_bias: float = 0.85,
    jitter_frac: float = 0.06,
    inset_frac: float | None = None,
    inset_frac_x: float = 0.28,
    inset_frac_y: float = 0.30,
    coverage: str = "center_band",
    band_half_frac: float = DEFAULT_A1_BAND_HALF_FRAC,
    rng: random.Random | None = None,
    block_id: str | None = None,
    image_w: int | None = None,
    image_h: int | None = None,
    image_edge_margin: float = DEFAULT_IMAGE_EDGE_MARGIN_PX,
    fragment_index: int | None = None,
) -> list[SampledPoint]:
    """Sample R points inside a *content* (text-ink) box.

    Default ``coverage="center_band"`` (A1): uniform in center ± ``band_half_frac``
    of box width/height (default 0.35).
    ``coverage="center_heavy"`` keeps the older tight-center anchors.
    ``coverage="corners"`` keeps corner/edge spread for later stages.
    ``coverage="diamond5"`` splits the inner-80% rectangle (side midpoints →
    diamond) into 5 regions and samples all 5 × 2 points.
    """
    if n < 1:
        return []
    if box.width() < 4 or box.height() < 4:
        return []

    rng = rng or random.Random()
    if inset_frac is not None:
        inset_frac_x = inset_frac_y = inset_frac

    def _accept(x: float, y: float) -> bool:
        if image_w is None or image_h is None:
            return True
        return not near_image_edge(x, y, image_w, image_h, margin=image_edge_margin)

    points: list[SampledPoint] = []

    if coverage in {"diamond5", "diamond"}:
        return sample_diamond5_points(
            box,
            n_regions=DEFAULT_DIAMOND5_N_REGIONS,
            points_per_region=DEFAULT_DIAMOND5_POINTS_PER_REGION,
            rng=rng,
            block_id=block_id,
            image_w=image_w,
            image_h=image_h,
            image_edge_margin=image_edge_margin,
            fragment_index=fragment_index,
        )

    if coverage in {"center_band", "a1"}:
        band = center_band_box(box, band_half_frac=band_half_frac)
        if band.width() < 2 or band.height() < 2:
            return []
        tries = 0
        while len(points) < n and tries < n * 40:
            tries += 1
            x = rng.uniform(band.x0, band.x1)
            y = rng.uniform(band.y0, band.y1)
            x, y = band.clamp_point(x, y, inset=0.5)
            if not _accept(x, y):
                continue
            points.append(
                SampledPoint(
                    x=x,
                    y=y,
                    kind="positive",
                    block_id=block_id,
                    region="center_band",
                    fragment_index=fragment_index,
                    fragment_bbox=box,
                )
            )
        return points

    if coverage in {"inner_area", "inner_80"}:
        band = inner_area_box(box)
        if band.width() < 2 or band.height() < 2:
            return []
        tries = 0
        while len(points) < n and tries < n * 40:
            tries += 1
            x = rng.uniform(band.x0, band.x1)
            y = rng.uniform(band.y0, band.y1)
            x, y = band.clamp_point(x, y, inset=0.5)
            if not _accept(x, y):
                continue
            points.append(
                SampledPoint(
                    x=x,
                    y=y,
                    kind="positive",
                    block_id=block_id,
                    region="inner_area",
                    fragment_index=fragment_index,
                    fragment_bbox=box,
                )
            )
        return points

    if coverage == "edge_band":
        tries = 0
        while len(points) < n and tries < n * 40:
            tries += 1
            got = _sample_edge_band_point(box, rng)
            if got is None:
                break
            x, y, region = got
            if not _accept(x, y):
                continue
            points.append(
                SampledPoint(
                    x=x,
                    y=y,
                    kind="positive",
                    block_id=block_id,
                    region=region,
                    fragment_index=fragment_index,
                    fragment_bbox=box,
                )
            )
        return points

    # Extreme corners: stay almost on the full ink box (tiny inset only).
    if coverage == "corner_extreme":
        inset_frac_x, inset_frac_y = 0.03, 0.04

    inner = _inner_box(box, inset_frac_x, inset_frac_y)
    w, h = inner.width(), inner.height()
    if w < 2 or h < 2:
        return []
    jx = max(w * jitter_frac, 0.5)
    jy = max(h * jitter_frac, 0.5)

    if coverage == "center":
        cx = (inner.x0 + inner.x1) / 2.0
        cy = (inner.y0 + inner.y1) / 2.0
        tries = 0
        while len(points) < n and tries < n * 40:
            tries += 1
            if rng.random() < center_bias:
                x, y = rng.gauss(cx, jx), rng.gauss(cy, jy)
                region = "center"
            else:
                x = rng.uniform(inner.x0, inner.x1)
                y = rng.uniform(inner.y0, inner.y1)
                region = "uniform"
            x, y = inner.clamp_point(x, y, inset=0.5)
            if not _accept(x, y):
                continue
            points.append(
                SampledPoint(
                    x=x,
                    y=y,
                    kind="positive",
                    block_id=block_id,
                    region=region,
                    fragment_index=fragment_index,
                    fragment_bbox=box,
                )
            )
        return points

    if coverage == "corner_extreme":
        anchors = list(_CORNER_EXTREME_ANCHORS)
        # Smaller jitter so points stay near the extreme anchors.
        jx = max(w * min(jitter_frac, 0.03), 0.3)
        jy = max(h * min(jitter_frac, 0.03), 0.3)
    elif coverage == "corners":
        anchors = list(_CORNER_ANCHORS)
    else:
        # center_heavy (legacy) and any unknown → safe center-ish
        anchors = list(_CENTER_HEAVY_ANCHORS)

    regions = list(anchors)
    rng.shuffle(regions)
    while len(regions) < n * 3:
        regions.extend(anchors)
        rng.shuffle(regions)

    ri = 0
    tries = 0
    while len(points) < n and tries < n * 40:
        tries += 1
        name, u, v = regions[ri % len(regions)]
        ri += 1
        x = inner.x0 + u * w + rng.uniform(-jx, jx)
        y = inner.y0 + v * h + rng.uniform(-jy, jy)
        x, y = inner.clamp_point(x, y, inset=0.5)
        if not _accept(x, y):
            continue
        points.append(
            SampledPoint(
                x=x,
                y=y,
                kind="positive",
                block_id=block_id,
                region=name,
                fragment_index=fragment_index,
                fragment_bbox=box,
            )
        )
    return points


def sample_points_in_fragments(
    fragments: Sequence[BBox],
    *,
    n: int = 3,
    coverage: str = "center_band",
    band_half_frac: float = DEFAULT_A1_BAND_HALF_FRAC,
    rng: random.Random | None = None,
    block_id: str | None = None,
    image_w: int | None = None,
    image_h: int | None = None,
    image_edge_margin: float = DEFAULT_IMAGE_EDGE_MARGIN_PX,
) -> list[SampledPoint]:
    """Sample points inside ink fragments (never in the gutter of a union AABB).

    Each point is drawn from one fragment chosen with probability ∝ area.
    ``diamond5`` picks one fragment and returns all 5 regions × 2 points on it.
    """
    rng = rng or random.Random()
    usable = [(i, b) for i, b in enumerate(fragments) if b.width() >= 4 and b.height() >= 4]
    if not usable or n < 1:
        return []
    weights = [max(b.area(), 1.0) for _, b in usable]
    if coverage in {"diamond5", "diamond"}:
        fi, box = rng.choices(usable, weights=weights, k=1)[0]
        return sample_diamond5_points(
            box,
            n_regions=DEFAULT_DIAMOND5_N_REGIONS,
            points_per_region=DEFAULT_DIAMOND5_POINTS_PER_REGION,
            rng=rng,
            block_id=block_id,
            image_w=image_w,
            image_h=image_h,
            image_edge_margin=image_edge_margin,
            fragment_index=fi,
        )
    out: list[SampledPoint] = []
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        pick = rng.choices(usable, weights=weights, k=1)[0]
        fi, box = pick
        got = sample_points_in_block(
            box,
            n=1,
            coverage=coverage,
            band_half_frac=band_half_frac,
            rng=rng,
            block_id=block_id,
            image_w=image_w,
            image_h=image_h,
            image_edge_margin=image_edge_margin,
            fragment_index=fi,
        )
        out.extend(got)
    return out


def sample_negative_points(
    image_w: int,
    image_h: int,
    blocks: Sequence[BBox],
    *,
    n: int = 4,
    margin: float = DEFAULT_IMAGE_EDGE_MARGIN_PX,
    clearance: float = DEFAULT_NEG_CLEARANCE_PX,
    max_tries: int = 800,
    rng: random.Random | None = None,
) -> list[SampledPoint]:
    """Sample empty-label points: off all text, and away from image borders.

    Does **not** hug the screenshot edge (that looked like too_hard in review).
    ``margin`` = min distance to image border; ``clearance`` = min distance to
    any block (after treating blocks as forbidden disks/rects).
    """
    rng = rng or random.Random()
    out: list[SampledPoint] = []
    if image_w <= 2 * margin + 4 or image_h <= 2 * margin + 4:
        return out

    expanded = [b.expand(clearance) for b in blocks]
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        x = rng.uniform(margin, image_w - margin)
        y = rng.uniform(margin, image_h - margin)
        if any(b.contains(x, y) for b in expanded):
            continue
        # Extra: keep a soft gap even if expand missed skinny slivers
        if blocks and _min_dist_outside_boxes(x, y, blocks) < clearance:
            continue
        out.append(SampledPoint(x=x, y=y, kind="negative", block_id=None, region="neg_clear"))
    return out


def sample_boundary_empty_points(
    image_w: int,
    image_h: int,
    blocks: Sequence[BBox],
    *,
    n: int = 4,
    max_gap: float = DEFAULT_BOUNDARY_MAX_GAP_PX,
    area_frac: float = DEFAULT_INNER_AREA_FRAC,
    margin: float = DEFAULT_IMAGE_EDGE_MARGIN_PX,
    max_tries: int = 1600,
    rng: random.Random | None = None,
) -> list[SampledPoint]:
    """Empty-label points in the junction of two nearby boxes, outside every full bbox.

    The inner-80%–100% ring is left unlabeled (SFT does not sample it; RL later).
    Only points strictly outside every unit's 100% bbox are empty_boundary.
    """
    rng = rng or random.Random()
    out: list[SampledPoint] = []
    usable = [b for b in blocks if b.width() >= 8 and b.height() >= 8]
    if len(usable) < 2:
        return out
    _ = area_frac  # kept for call-site compat; ring is no longer a negative.
    pairs: list[tuple[BBox, BBox, BBox]] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            gap = facing_gap_px(usable[i], usable[j])
            if gap is None or gap > max_gap:
                continue
            strip = gap_strip(usable[i], usable[j])
            if strip is None or strip.area() < 4:
                strip = usable[i].expand(6)
            sample_box = strip.expand(8)
            if sample_box.width() < 2 or sample_box.height() < 2:
                continue
            pairs.append((usable[i], usable[j], sample_box))
    if not pairs:
        return out

    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        _a, _b, zone = rng.choice(pairs)
        x = rng.uniform(zone.x0, zone.x1)
        y = rng.uniform(zone.y0, zone.y1)
        if near_image_edge(x, y, image_w, image_h, margin=margin):
            continue
        if any(b.contains(x, y) for b in usable):
            continue
        out.append(
            SampledPoint(x=x, y=y, kind="negative", block_id=None, region="neg_boundary")
        )
    return out


def points_per_block_range(r_min: int = 2, r_max: int = 5, rng: random.Random | None = None) -> int:
    rng = rng or random.Random()
    return rng.randint(r_min, r_max)

"""Sample interest points inside block bboxes + negative (empty-label) points."""

from __future__ import annotations

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

    def clamp_point(self, x: float, y: float, *, inset: float = 1.0) -> tuple[float, float]:
        x = min(max(x, self.x0 + inset), self.x1 - inset)
        y = min(max(y, self.y0 + inset), self.y1 - inset)
        return x, y

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class SampledPoint:
    x: float
    y: float
    kind: str  # "positive" | "negative"
    block_id: str | None = None
    region: str | None = None  # e.g. "nw", "center", "east"


# Relative (u,v) in [0,1]^2 for stratified coverage of a (usually wide, short) text box.
# Corners first so even small R hits extremities of long horizontal lines.
_REGION_ANCHORS: list[tuple[str, float, float]] = [
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


def sample_points_in_block(
    box: BBox,
    *,
    n: int = 3,
    # Kept for API compat; ignored when coverage="corners" (default).
    center_bias: float = 0.15,
    jitter_frac: float = 0.08,
    inset_frac: float | None = None,
    inset_frac_x: float = 0.06,
    inset_frac_y: float = 0.12,
    coverage: str = "corners",
    rng: random.Random | None = None,
    block_id: str | None = None,
) -> list[SampledPoint]:
    """Sample R points inside a *content* (text-ink) box.

    Default ``coverage="corners"`` spreads points across corners / edges / mid
    of typically wide-and-short text lines, instead of clustering at the center.
    """
    if n < 1:
        return []
    if box.width() < 4 or box.height() < 4:
        return []

    rng = rng or random.Random()
    if inset_frac is not None:
        inset_frac_x = inset_frac_y = inset_frac

    inner = _inner_box(box, inset_frac_x, inset_frac_y)
    w, h = inner.width(), inner.height()
    jx = max(w * jitter_frac, 0.6)
    jy = max(h * jitter_frac, 0.6)

    points: list[SampledPoint] = []

    if coverage == "center":
        # Legacy-ish: mostly center (not recommended for POINT)
        cx = (inner.x0 + inner.x1) / 2.0
        cy = (inner.y0 + inner.y1) / 2.0
        for _ in range(n):
            if rng.random() < center_bias:
                x, y = rng.gauss(cx, jx), rng.gauss(cy, jy)
                region = "center"
            else:
                x = rng.uniform(inner.x0, inner.x1)
                y = rng.uniform(inner.y0, inner.y1)
                region = "uniform"
            x, y = inner.clamp_point(x, y, inset=0.5)
            points.append(
                SampledPoint(x=x, y=y, kind="positive", block_id=block_id, region=region)
            )
        return points

    # Stratified: cycle a shuffled region list so R=2..5 still hits corners.
    regions = list(_REGION_ANCHORS)
    rng.shuffle(regions)
    # Repeat / extend if n > len(regions)
    while len(regions) < n:
        regions.extend(_REGION_ANCHORS)
        rng.shuffle(regions)

    for i in range(n):
        name, u, v = regions[i]
        # Extra jitter; keep inside inner box
        x = inner.x0 + u * w + rng.uniform(-jx, jx)
        y = inner.y0 + v * h + rng.uniform(-jy, jy)
        x, y = inner.clamp_point(x, y, inset=0.5)
        points.append(
            SampledPoint(x=x, y=y, kind="positive", block_id=block_id, region=name)
        )
    return points


def sample_negative_points(
    image_w: int,
    image_h: int,
    blocks: Sequence[BBox],
    *,
    n: int = 4,
    margin: float = 8.0,
    max_tries: int = 200,
    rng: random.Random | None = None,
) -> list[SampledPoint]:
    """Sample points outside all block boxes (page edge / gutter / chrome)."""
    rng = rng or random.Random()
    out: list[SampledPoint] = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        mode = rng.choice(["edge", "edge", "uniform"])
        if mode == "edge":
            side = rng.choice(["top", "bottom", "left", "right"])
            if side == "top":
                x, y = rng.uniform(0, image_w), rng.uniform(0, max(margin, 1))
            elif side == "bottom":
                x, y = rng.uniform(0, image_w), rng.uniform(image_h - margin, image_h)
            elif side == "left":
                x, y = rng.uniform(0, max(margin, 1)), rng.uniform(0, image_h)
            else:
                x, y = rng.uniform(image_w - margin, image_w), rng.uniform(0, image_h)
        else:
            x = rng.uniform(0, image_w)
            y = rng.uniform(0, image_h)

        if any(b.contains(x, y, margin=-2.0) for b in blocks):
            continue
        out.append(SampledPoint(x=x, y=y, kind="negative", block_id=None, region="neg"))
    return out


def points_per_block_range(r_min: int = 2, r_max: int = 5, rng: random.Random | None = None) -> int:
    rng = rng or random.Random()
    return rng.randint(r_min, r_max)

"""Select in-frame blocks for pool emission."""

from __future__ import annotations

import random

from point_ocr.a2_labels import block_fully_in_frame, classify_a2_kind
from point_ocr.build_point import BlockAnno
from point_ocr.filter_qa import bucket_for_block, filter_block_label

_EMPTY_KINDS = frozenset({"table", "image", "chart"})
OCC_MIN = 0.25
OCC_MAX = 0.70


def is_special_block(block: BlockAnno) -> bool:
    """Tables / images / charts are scenery: POINT must return empty.

    Formulas and code are normal text targets (including `$` inside prose).
    """
    extra = block.extra or {}
    kind = classify_a2_kind(extra, extra.get("tag"), block.markdown or "")
    if kind in _EMPTY_KINDS:
        return True
    bucket = bucket_for_block(block.markdown or "", tag=extra.get("tag"))
    return bucket == "table"


def occupancy_in_train_range(occ: float | None) -> bool:
    """Q1 training gate. ``None`` keeps legacy rows that never recorded occupancy."""
    if occ is None:
        return True
    return OCC_MIN <= float(occ) <= OCC_MAX


def large_blocks(
    blocks: list[BlockAnno],
    page_area: float,
    *,
    image_w: int,
    image_h: int,
    min_area: float = 0.015,
    min_chars: int = 40,
) -> list[BlockAnno]:
    kept: list[BlockAnno] = []
    for b in blocks:
        if is_special_block(b):
            continue
        fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
        if not fr.keep:
            continue
        if not block_fully_in_frame(b, image_w, image_h):
            continue
        if page_area > 0 and b.ink_area() / page_area < min_area:
            continue
        if len((b.markdown or "").strip()) < min_chars:
            continue
        kept.append(b)
    kept.sort(key=lambda b: b.ink_area(), reverse=True)
    return kept


def text_occupancy_frac(
    blocks: list[BlockAnno],
    image_w: int,
    image_h: int,
    *,
    include_special: bool = False,
    scale: int = 8,
) -> float:
    """Union of text-ink rectangles / page area (avoids overlapping bbox double-count)."""
    from PIL import Image, ImageDraw

    if image_w <= 0 or image_h <= 0:
        return 0.0
    mw = max(1, int(image_w) // scale)
    mh = max(1, int(image_h) // scale)
    sx = mw / float(image_w)
    sy = mh / float(image_h)
    im = Image.new("1", (mw, mh), 0)
    dr = ImageDraw.Draw(im)
    for b in blocks:
        if not include_special and is_special_block(b):
            continue
        for r in b.ink_rects():
            x0 = max(0, int(r.x0 * sx))
            y0 = max(0, int(r.y0 * sy))
            x1 = min(mw, int(r.x1 * sx))
            y1 = min(mh, int(r.y1 * sy))
            if x1 > x0 and y1 > y0:
                dr.rectangle([x0, y0, x1 - 1, y1 - 1], fill=1)
    pixels = im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata()
    return float(sum(pixels)) / float(mw * mh)


def pick_spread(blocks: list[BlockAnno], n: int, rng: random.Random) -> list[BlockAnno]:
    """Avoid always taking reading-order first blocks."""
    if len(blocks) <= n:
        return list(blocks)
    by_y = sorted(blocks, key=lambda b: b.bbox.y0)
    skip = max(1, len(by_y) // 5)
    pool = by_y[skip:] or by_y
    large = sorted(pool, key=lambda b: b.ink_area(), reverse=True)
    chosen: list[BlockAnno] = []
    for b in large:
        if b not in chosen:
            chosen.append(b)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        rest = [b for b in pool if b not in chosen]
        rng.shuffle(rest)
        chosen.extend(rest[: n - len(chosen)])
    return chosen


def multi_frag_blocks(
    blocks: list[BlockAnno],
    *,
    image_w: int,
    image_h: int,
    min_chars: int = 30,
) -> list[BlockAnno]:
    multi = [
        b
        for b in blocks
        if not is_special_block(b)
        and len(b.ink_rects()) > 1
        and len((b.markdown or "").strip()) >= min_chars
        and block_fully_in_frame(b, image_w, image_h)
    ]
    multi.sort(key=lambda b: b.ink_area(), reverse=True)
    return multi

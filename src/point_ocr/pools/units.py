"""Collapse related short texts into one occupancy / hit bbox.

A short-text unit (heading+list, title+byline, tagged semantic group) is one
bbox. The gutter between its members is inside the unit — not an empty-boundary
negative. Junctions *between* that unit and a different block stay empty.
"""

from __future__ import annotations

from dataclasses import dataclass

from point_ocr.a2_labels import SemanticGroup, parse_semantic_groups
from point_ocr.build_point import BlockAnno
from point_ocr.pools.select import is_special_block
from point_ocr.sample_points import (
    DEFAULT_BOUNDARY_MAX_GAP_PX,
    BBox,
    facing_gap_px,
    union_bbox,
)

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_ITEM_TAGS = frozenset({"li", "dt", "dd"})
_DEK_TAGS = frozenset({"p", "div", "span"})
_SHORT_CHARS = 80
_CLUSTER_GAP_PX = min(36.0, DEFAULT_BOUNDARY_MAX_GAP_PX)
_MIN_X_OVERLAP_FRAC = 0.35


@dataclass(frozen=True)
class TextUnit:
    """One POINT-able semantic unit: a single block or a collapsed short group."""

    unit_id: str
    bbox: BBox
    member_ids: tuple[str, ...]
    markdown: str
    kind: str  # "block" | "semantic_group" | "short_cluster"


def _tag(block: BlockAnno) -> str:
    return str((block.extra or {}).get("tag") or "").lower()


def _md(block: BlockAnno) -> str:
    return (block.markdown or "").strip()


def _explicit_gid(block: BlockAnno) -> str | None:
    g = (block.extra or {}).get("semantic_group")
    if g in (None, "", "null", "None"):
        return None
    return str(g)


def _role(block: BlockAnno) -> str | None:
    if is_special_block(block) or not _md(block):
        return None
    tag = _tag(block)
    if tag in _HEADING_TAGS and len(_md(block)) < 120:
        return "h"
    if tag in _ITEM_TAGS:
        return "li"
    if tag in _DEK_TAGS and len(_md(block)) <= _SHORT_CHARS and block.bbox.height() < 72:
        return "dek"
    return None


def _x_overlap_frac(a: BBox, b: BBox) -> float:
    ov = min(a.x1, b.x1) - max(a.x0, b.x0)
    denom = max(1.0, min(a.width(), b.width()))
    return float(ov) / denom


def _can_append(cluster: list[BlockAnno], nxt: BlockAnno) -> bool:
    prev = _role(cluster[-1])
    role = _role(nxt)
    if prev is None or role is None:
        return False
    gap = facing_gap_px(cluster[-1].bbox, nxt.bbox)
    if gap is None or gap > _CLUSTER_GAP_PX:
        return False
    if _x_overlap_frac(cluster[-1].bbox, nxt.bbox) < _MIN_X_OVERLAP_FRAC:
        return False
    if role == "li":
        return prev in {"h", "dek", "li"}
    if role == "dek":
        return prev == "h"
    if role == "h":
        return prev == "dek"
    return False


def _keep_cluster(cluster: list[BlockAnno]) -> bool:
    if len(cluster) < 2:
        return False
    roles = {_role(b) for b in cluster}
    if "li" in roles:
        return True
    return roles == {"h", "dek"}


def _ink_union(members: list[BlockAnno]) -> BBox | None:
    boxes = [r for m in members for r in m.ink_rects()]
    return union_bbox(boxes)


def _join_md(members: list[BlockAnno]) -> str:
    return "\n\n".join(_md(m) for m in members if _md(m))


def infer_short_clusters(blocks: list[BlockAnno]) -> list[list[BlockAnno]]:
    """Untagged heading+list / title+byline stacks (explicit groups excluded)."""
    cands = [b for b in blocks if _role(b) is not None and _explicit_gid(b) is None]
    ordered = sorted(cands, key=lambda b: (b.bbox.y0, b.bbox.x0, b.block_id))
    clusters: list[list[BlockAnno]] = []
    used: set[str] = set()
    for seed in ordered:
        if seed.block_id in used:
            continue
        cluster = [seed]
        used.add(seed.block_id)
        for nxt in ordered:
            if nxt.block_id in used:
                continue
            if _can_append(cluster, nxt):
                cluster.append(nxt)
                used.add(nxt.block_id)
        cluster.sort(key=lambda b: (b.bbox.y0, b.bbox.x0, b.block_id))
        if _keep_cluster(cluster):
            clusters.append(cluster)
        else:
            for b in cluster:
                used.discard(b.block_id)
    return clusters


def collect_text_units(
    blocks: list[BlockAnno],
    page_html: str | None = None,
) -> list[TextUnit]:
    """Member blocks of a short unit are replaced by that unit's union bbox."""
    by_id = {b.block_id: b for b in blocks}
    assigned: set[str] = set()
    units: list[TextUnit] = []

    groups: list[SemanticGroup] = parse_semantic_groups(page_html) if page_html else []
    for g in groups:
        members = [by_id[mid] for mid in g.member_ids if mid in by_id]
        box = _ink_union(members)
        if not members or box is None:
            continue
        units.append(
            TextUnit(
                unit_id=f"sg:{g.group_id}",
                bbox=box,
                member_ids=tuple(m.block_id for m in members),
                markdown=(g.markdown or "").strip() or _join_md(members),
                kind="semantic_group",
            )
        )
        assigned.update(m.block_id for m in members)

    rest = [b for b in blocks if b.block_id not in assigned]
    for i, cluster in enumerate(infer_short_clusters(rest)):
        box = _ink_union(cluster)
        if box is None:
            continue
        units.append(
            TextUnit(
                unit_id=f"sc:{cluster[0].block_id}:{i}",
                bbox=box,
                member_ids=tuple(m.block_id for m in cluster),
                markdown=_join_md(cluster),
                kind="short_cluster",
            )
        )
        assigned.update(m.block_id for m in cluster)

    for b in blocks:
        if b.block_id in assigned or is_special_block(b):
            continue
        box = union_bbox(b.ink_rects()) or b.bbox
        units.append(
            TextUnit(
                unit_id=b.block_id,
                bbox=box,
                member_ids=(b.block_id,),
                markdown=_md(b),
                kind="block",
            )
        )
    return units


def occupancy_boxes(
    blocks: list[BlockAnno],
    page_html: str | None = None,
) -> list[BBox]:
    """Boxes that count as text occupancy (cluster unions + leftover blocks)."""
    return [u.bbox for u in collect_text_units(blocks, page_html)]


def clustered_member_ids(
    blocks: list[BlockAnno],
    page_html: str | None = None,
) -> set[str]:
    """Block ids that belong to a multi-member short unit (not singleton core)."""
    out: set[str] = set()
    for u in collect_text_units(blocks, page_html):
        if len(u.member_ids) > 1:
            out.update(u.member_ids)
    return out

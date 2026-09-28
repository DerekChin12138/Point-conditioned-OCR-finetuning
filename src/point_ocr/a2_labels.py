"""A2 label helpers: semantic groups, table HTML, image/chart empty targets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup, Tag

from point_ocr.html_to_md import html_fragment_to_markdown, normalize_markdown


_DENY_KINDS = frozenset({"image", "chart"})


@dataclass
class SemanticGroup:
    group_id: str
    kind: str
    member_ids: list[str]  # DOM order
    seed_ids: list[str]
    body_ids: list[str]
    markdown: str
    translation: str = ""


def table_element_to_ovis_html(table: Tag) -> str:
    """Serialize a <table> to a compact Ovis-style HTML string."""
    # Drop id/class/style noise; keep structure + text.
    clone = BeautifulSoup(str(table), "lxml").find("table")
    if clone is None:
        return ""
    for el in [clone, *clone.find_all(True)]:
        for k in list(el.attrs.keys()):
            if k.startswith("data-") or k in {"class", "style", "id", "role"}:
                del el.attrs[k]
    html = str(clone)
    html = re.sub(r"\s+", " ", html)
    html = html.replace("> <", "><")
    return normalize_markdown(html)


def classify_a2_kind(extra: dict[str, Any] | None, tag: str | None, markdown: str) -> str:
    """Return a2 kind: formula|code|image|table|text."""
    ex = extra or {}
    kind = str(ex.get("a2_kind") or ex.get("data-a2-kind") or "").lower().strip()
    if kind in {"formula", "code", "image", "chart", "table"}:
        return kind
    t = (tag or str(ex.get("tag") or "")).lower()
    md = (markdown or "").strip()
    if t == "table" or (ex.get("html") and "<table" in str(ex.get("html")).lower() and t == "table"):
        return "table"
    if t in {"pre", "code"} or md.startswith("```"):
        return "code"
    if t == "img" or kind == "image":
        return "image"
    if ex.get("data-latex") or "$" in md[:80] and ("\\" in md or "sum" in md or "frac" in md):
        # Heuristic for formula blocks
        if t in {"div", "span"} and ("latex" in str(ex).lower() or "formula" in str(ex.get("html") or "").lower()):
            return "formula"
        if ex.get("data-latex"):
            return "formula"
    latex = ex.get("data-latex")
    if latex:
        return "formula"
    return "text"


def resolve_a2_target(
    *,
    kind: str,
    markdown: str,
    html_fragment: str | None = None,
) -> str:
    """Map block kind → training target under Q1 / a2_v3 rules.

    Images/charts → empty. Tables keep HTML when selected as targets (Q1 empties
    them via ``is_special_block`` instead). Formulas and code are transcribed.
    """
    if kind in _DENY_KINDS:
        return ""
    if kind == "table":
        if html_fragment:
            soup = BeautifulSoup(html_fragment, "lxml")
            table = soup.find("table")
            if table is not None:
                return table_element_to_ovis_html(table)
        if "<table" in (markdown or "").lower():
            return normalize_markdown(markdown)
        return markdown
    return markdown


def parse_semantic_groups(page_html: str) -> list[SemanticGroup]:
    """Find ``data-semantic-group`` sections and build aggregated Markdown GT."""
    soup = BeautifulSoup(page_html, "lxml")
    groups: list[SemanticGroup] = []
    for sec in soup.find_all(attrs={"data-semantic-group": True}):
        if not isinstance(sec, Tag):
            continue
        gid = str(sec.get("data-semantic-group"))
        kind = str(sec.get("data-group-kind") or "")
        member_ids: list[str] = []
        seed_ids: list[str] = []
        body_ids: list[str] = []
        zh_parts: list[str] = []
        for el in sec.find_all(attrs={"data-block-id": True}):
            if not isinstance(el, Tag):
                continue
            parent_g = el.find_parent(attrs={"data-semantic-group": True})
            if parent_g is not sec:
                continue
            bid = str(el.get("data-block-id"))
            role = str(el.get("data-role") or "member").lower()
            member_ids.append(bid)
            if role == "seed":
                seed_ids.append(bid)
            else:
                body_ids.append(bid)
            zh = str(el.get("data-mt-zh") or "").strip()
            if zh:
                zh_parts.append(zh)
        if not member_ids:
            continue
        clone = BeautifulSoup(str(sec), "lxml")
        root = clone.find(attrs={"data-semantic-group": True}) or clone.body
        if isinstance(root, Tag):
            for deco in root.select(".kind"):
                deco.decompose()
            md = html_fragment_to_markdown(str(root))
        else:
            md = ""
        groups.append(
            SemanticGroup(
                group_id=gid,
                kind=kind,
                member_ids=member_ids,
                seed_ids=seed_ids or member_ids[:1],
                body_ids=body_ids or member_ids[1:],
                markdown=md,
                translation="\n\n".join(zh_parts),
            )
        )
    return groups


def is_block_visibly_clear(
    extra: dict[str, Any] | None,
    *,
    min_visible_frac: float = 0.9,
) -> bool:
    """True if render-time visibility check says the block is mostly unoccluded.

    Missing ``visible_frac`` (legacy renders) → treat as clear.
    """
    if not extra:
        return True
    vf = extra.get("visible_frac")
    if vf is None:
        return True
    try:
        return float(vf) >= float(min_visible_frac)
    except (TypeError, ValueError):
        return True


def block_fully_in_frame(
    block: Any,
    image_w: int,
    image_h: int,
    *,
    min_visible_frac: float = 0.98,
    min_in_viewport_frac: float = 0.98,
    max_outside_px: float = 2.0,
) -> bool:
    """True only when block ink is essentially entirely inside the screenshot.

    Used for focused-window captures where ``full_page=False`` can crop HTML
    blocks. Partial crops must not keep full-block Markdown GT.
    """
    if image_w <= 2 or image_h <= 2:
        return False

    rects = None
    if hasattr(block, "ink_rects"):
        try:
            rects = list(block.ink_rects())
        except Exception:
            rects = None
    if not rects:
        bbox = getattr(block, "bbox", None)
        if bbox is None:
            return False
        rects = [bbox]

    for r in rects:
        x0 = float(getattr(r, "x0", r[0] if isinstance(r, (list, tuple)) else 0))
        y0 = float(getattr(r, "y0", r[1] if isinstance(r, (list, tuple)) else 0))
        x1 = float(getattr(r, "x1", r[2] if isinstance(r, (list, tuple)) else 0))
        y1 = float(getattr(r, "y1", r[3] if isinstance(r, (list, tuple)) else 0))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return False
        if x0 < -max_outside_px or y0 < -max_outside_px:
            return False
        if x1 > float(image_w) + max_outside_px or y1 > float(image_h) + max_outside_px:
            return False

    extra = getattr(block, "extra", None) or {}
    # Union-bbox visible_frac samples the gutter between column fragments, so a
    # fully on-screen wrapped paragraph scores ~0.2–0.4. Per-rect geometry (and
    # in_viewport_frac) already encode crop; skip vf when there are fragments.
    if len(rects) <= 1:
        vf = extra.get("visible_frac")
        if vf is not None:
            try:
                if float(vf) < float(min_visible_frac):
                    return False
            except (TypeError, ValueError):
                pass
    ivf = extra.get("in_viewport_frac")
    if ivf is not None:
        try:
            if float(ivf) < float(min_in_viewport_frac):
                return False
        except (TypeError, ValueError):
            pass
    return True


def is_focused_window_block(extra: dict[str, Any] | None) -> bool:
    if not extra:
        return False
    return bool(extra.get("window_focused"))


def build_semantic_group_map(page_html: str) -> dict[str, SemanticGroup]:
    """Map each member block_id → its SemanticGroup."""
    out: dict[str, SemanticGroup] = {}
    for g in parse_semantic_groups(page_html):
        for mid in g.member_ids:
            out[mid] = g
    return out


def find_enclosing_table_html(page_html: str, block_id: str) -> str | None:
    """If ``block_id`` is a table or any cell inside one, return whole-table Ovis HTML."""
    soup = BeautifulSoup(page_html, "lxml")
    el = soup.find(attrs={"data-block-id": block_id})
    if el is None or not isinstance(el, Tag):
        return None
    table = el if el.name == "table" else el.find_parent("table")
    if table is None or not isinstance(table, Tag):
        return None
    return table_element_to_ovis_html(table)


def rewrite_block_for_a2(
    block: Any,
    page_html: str | None,
    group_map: dict[str, SemanticGroup] | None = None,
) -> Any:
    """Apply A2 target rules.

    Priority: table (full HTML) → semantic-group aggregate → image/chart empty
    → otherwise keep markdown (including formulas and code).
    """
    from point_ocr.build_point import BlockAnno

    extra = dict(getattr(block, "extra", None) or {})
    tag = str(extra.get("tag") or "")
    md = getattr(block, "markdown", "") or ""
    bid = str(getattr(block, "block_id", ""))
    kind = classify_a2_kind(extra, tag, md)
    if extra.get("a2_kind") and extra.get("a2_kind") not in {"text", "semantic_group"}:
        kind = str(extra["a2_kind"])

    table_html = None
    if page_html:
        table_html = find_enclosing_table_html(page_html, bid)
    if table_html:
        kind = "table"
        target = table_html
        extra = {**extra, "a2_kind": kind}
    else:
        gmap = group_map
        if gmap is None and page_html:
            gmap = build_semantic_group_map(page_html)
        g = gmap.get(bid) if gmap else None
        if g is not None and g.markdown.strip():
            kind = "semantic_group"
            target = g.markdown
            role = "seed" if bid in g.seed_ids else "body"
            extra = {
                **extra,
                "a2_kind": kind,
                "group_id": g.group_id,
                "group_kind": g.kind,
                "group_role": role,
            }
        else:
            target = resolve_a2_target(
                kind=kind,
                markdown=md,
                html_fragment=extra.get("html"),
            )
            extra = {**extra, "a2_kind": kind}

    return BlockAnno(
        block_id=block.block_id,
        bbox=block.bbox,
        markdown=target,
        rects=list(getattr(block, "rects", None) or []),
        extra=extra,
    )


def select_desktop_positive_blocks(
    blocks: list[Any],
    *,
    max_n: int = 3,
    min_visible_frac: float = 0.9,
    focused_frac: float = 0.75,
) -> list[Any]:
    """Prefer fully-visible blocks on the focused window; never keep occluded text.

    ``blocks`` are ``BlockAnno``-like objects with ``.extra`` and ``.ink_area()``.
    """
    clear = [b for b in blocks if is_block_visibly_clear(getattr(b, "extra", None), min_visible_frac=min_visible_frac)]
    focused = [b for b in clear if is_focused_window_block(getattr(b, "extra", None))]
    other = [b for b in clear if b not in focused]
    focused.sort(key=lambda b: b.ink_area(), reverse=True)
    other.sort(key=lambda b: b.ink_area(), reverse=True)
    n_focus = max(1, int(round(max_n * focused_frac))) if focused else 0
    n_focus = min(n_focus, max_n, len(focused))
    out = focused[:n_focus]
    remain = max_n - len(out)
    if remain > 0:
        out.extend(other[:remain])
    if not out and other:
        out = other[:max_n]
    elif not out and focused:
        out = focused[:max_n]
    return out

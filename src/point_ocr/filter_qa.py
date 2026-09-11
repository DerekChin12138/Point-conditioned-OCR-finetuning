"""Conservative filtering + bucketed QA helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from point_ocr.metrics import edit_similarity, normalize_text


_REPEAT_RE = re.compile(r"(.{8,80}?)\1{4,}", re.S)


@dataclass
class FilterResult:
    keep: bool
    reason: str
    bucket: str


def bucket_for_block(markdown: str, tag: str | None = None) -> str:
    t = (tag or "").lower()
    md = markdown.strip()
    if not md:
        return "empty"
    if t in {"table", "td", "th"} or "<table" in md.lower():
        return "table"
    if "$" in md or "\\frac" in md or "\\sum" in md:
        return "formula"
    if md.startswith("```") or t in {"pre", "code"}:
        return "code"
    if md.startswith("#"):
        return "heading"
    if len(md) < 40:
        return "short"
    if len(md) > 800:
        return "long"
    return "paragraph"


def filter_block_label(markdown: str, *, tag: str | None = None) -> FilterResult:
    """Drop obviously bad GT labels before training."""
    bucket = bucket_for_block(markdown, tag)
    md = markdown.strip()
    if bucket == "empty":
        return FilterResult(False, "empty_label", bucket)
    if _REPEAT_RE.search(md):
        return FilterResult(False, "repetition", bucket)
    if md.count("�") > 0:
        return FilterResult(False, "replacement_char", bucket)
    # Extremely unbalanced HTML
    if md.count("<") > 20 and md.count("<") != md.count(">"):
        return FilterResult(False, "broken_html", bucket)
    return FilterResult(True, "ok", bucket)


def filter_pair_consistency(
    pred_or_alt: str,
    gt: str,
    *,
    min_sim: float = 0.35,
) -> FilterResult:
    """Optional: drop pairs where an independent check is far from GT."""
    if not normalize_text(gt):
        return FilterResult(False, "empty_gt", "empty")
    sim = edit_similarity(pred_or_alt, gt)
    if sim < min_sim:
        return FilterResult(False, f"low_consistency:{sim:.2f}", bucket_for_block(gt))
    return FilterResult(True, "ok", bucket_for_block(gt))

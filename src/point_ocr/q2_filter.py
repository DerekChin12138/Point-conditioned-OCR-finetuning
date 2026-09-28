"""Rules for removing misleading Q2 negative samples.

Two negative pools were found to inject contradictory or mislabeled supervision:

``empty_special``
    Markers placed inside a table/chart.  Q2 filled the table templates with
    OPUS prose, so ~23% of these "table" cells contain full sentences that look
    exactly like the paragraph blocks ``core_inner`` rewards outputting.  The
    label (empty) contradicts a visually identical positive -> drop prose cells.

``empty_clear`` / ``neg_clear``
    ``neg_clear`` means "clear of all ink", but the ink-avoidance boxes never
    included the chrome bands, so ~9% of these points sit *inside* a text block
    and ~15% sit on chrome text.  Points on ink are mislabeled -> drop; points on
    chrome are legitimate hard negatives -> relabel to ``chrome_*``.

All geometry is converted from CSS px to image px with ``device_scale_factor``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Q2FilterConfig:
    max_special_chars: int = 120
    max_special_words: int = 12
    drop_clear_on_ink: bool = True
    drop_chrome_clear: bool = False
    drop_chrome_mislabel: bool = True
    relabel_clear_on_chrome: bool = True


@dataclass
class FilterVerdict:
    drop: bool
    reason: str = ""
    new_region: str | None = None


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class PageGeometry:
    """Cached per-pool page geometry (ink rects + chrome bands)."""

    def __init__(self, pools_root: Path):
        self.pools_root = Path(pools_root)
        self._blocks: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._chrome: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def blocks(self, pool_id: str, page_id: str) -> list[dict[str, Any]]:
        key = (pool_id, page_id)
        if key not in self._blocks:
            p = self.pools_root / pool_id / "renders" / f"{page_id}.blocks.json"
            self._blocks[key] = _read_json(p, []) if p.is_file() else []
        return self._blocks[key]

    def chrome(self, pool_id: str, page_id: str) -> list[dict[str, Any]]:
        key = (pool_id, page_id)
        if key not in self._chrome:
            p = self.pools_root / pool_id / "renders" / f"{page_id}.chrome.json"
            self._chrome[key] = _read_json(p, []) if p.is_file() else []
        return self._chrome[key]

    def ink_rects(self, pool_id: str, page_id: str, scale: float) -> list[list[float]]:
        rects: list[list[float]] = []
        for b in self.blocks(pool_id, page_id):
            for r in b.get("rects") or [b.get("bbox")]:
                if r:
                    rects.append([float(v) * scale for v in r])
        return rects

    def chrome_bands(self, pool_id: str, page_id: str, scale: float) -> list[tuple[str, list[float]]]:
        out: list[tuple[str, list[float]]] = []
        for row in self.chrome(pool_id, page_id):
            bb = row.get("bbox")
            if not bb:
                continue
            out.append((str(row.get("band") or "head"), [float(v) * scale for v in bb]))
        return out

    def special_markdown(self, pool_id: str, page_id: str, block_id: str | None) -> str | None:
        if not block_id:
            return None
        for b in self.blocks(pool_id, page_id):
            if b.get("id") == block_id:
                return str(b.get("markdown") or "")
        return None


def _inside(x: float, y: float, rect: list[float]) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def classify_negative(
    meta: dict[str, Any],
    geometry: PageGeometry,
    *,
    config: Q2FilterConfig | None = None,
) -> FilterVerdict:
    """Decide whether one *negative* row is misleading.  Positives are never dropped."""
    cfg = config or Q2FilterConfig()
    if not meta.get("is_negative") and str(meta.get("target") or "").strip():
        return FilterVerdict(drop=False)
    pool_id = str(meta.get("pool_id") or "")
    page_id = str(meta.get("page_id") or "")
    region = str(meta.get("region") or "")
    point = meta.get("point")
    if not pool_id or not page_id or not point:
        return FilterVerdict(drop=False)
    scale = float(meta.get("device_scale_factor") or 1.0)
    x, y = float(point[0]), float(point[1])

    if region.startswith("special_"):
        md = geometry.special_markdown(pool_id, page_id, meta.get("special_block_id"))
        if md is None:
            return FilterVerdict(drop=True, reason="special_block_missing")
        if len(md) > cfg.max_special_chars or len(md.split()) >= cfg.max_special_words:
            return FilterVerdict(drop=True, reason="special_prose")

    if region in {"neg_clear", "neg_boundary", ""}:
        if cfg.drop_clear_on_ink and any(
            _inside(x, y, r) for r in geometry.ink_rects(pool_id, page_id, scale)
        ):
            return FilterVerdict(drop=True, reason="clear_on_ink")
        if cfg.relabel_clear_on_chrome or cfg.drop_chrome_clear:
            for band, rect in geometry.chrome_bands(pool_id, page_id, scale):
                if _inside(x, y, rect):
                    if cfg.drop_chrome_clear:
                        return FilterVerdict(drop=True, reason="clear_on_chrome")
                    return FilterVerdict(drop=False, reason="clear_on_chrome", new_region=f"chrome_{band}")

    if region.startswith("chrome_") and cfg.drop_chrome_mislabel:
        if not any(_inside(x, y, rect) for _, rect in geometry.chrome_bands(pool_id, page_id, scale)):
            return FilterVerdict(drop=True, reason="chrome_mislabel")

    return FilterVerdict(drop=False)


def is_negative_row(row: dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    if meta.get("is_negative"):
        return True
    msgs = row.get("messages") or []
    target = str(msgs[-1].get("content") or "") if msgs else ""
    return not target.strip()

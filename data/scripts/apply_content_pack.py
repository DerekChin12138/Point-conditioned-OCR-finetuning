#!/usr/bin/env python3
"""Fill a template from a shared content pool + dense random layout CSS variants.

Slot routing by tag / id heuristics → headings|paragraphs|list_items|…
Also injects extra paragraphs into main content areas so pages look fuller.

Example:
  uv run python data/scripts/apply_content_pack.py \\
    --template data/synth/templates/01_article_twocol.html \\
    --pool data/synth/content_pools/a1_pool.json \\
    --out data/synth/filled/01_article_twocol__v0.html \\
    --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]

# Dense-first layout recipes (typography + packing).
LAYOUT_DENSE = [
    {
        "name": "serif_dense",
        "css": """
html, body { margin:0 !important; min-height:100vh !important; }
body { font-family: Georgia, "Times New Roman", serif !important; font-size: 14px !important; line-height: 1.45 !important; }
h1 { font-size: 26px !important; margin: 0 0 8px !important; }
h2 { font-size: 17px !important; margin: 14px 0 6px !important; }
.article, .main, article, .content { padding: 16px 18px 28px !important; }
p { margin: 0 0 8px !important; }
.shell, .layout { min-height: 100vh !important; }
.two { column-gap: 18px !important; }
""",
    },
    {
        "name": "sans_packed",
        "css": """
html, body { margin:0 !important; min-height:100vh !important; }
body { font-family: system-ui, "Segoe UI", sans-serif !important; font-size: 13px !important; line-height: 1.35 !important; }
h1 { font-size: 22px !important; margin: 0 0 6px !important; }
h2 { font-size: 15px !important; margin: 12px 0 4px !important; }
.article, .main, article, .content { padding: 10px 12px 20px !important; }
p { margin: 0 0 6px !important; }
.shell, .layout { min-height: 100vh !important; }
.rail a, .nav a { padding: 4px 6px !important; }
""",
    },
    {
        "name": "zh_packed",
        "css": """
html, body { margin:0 !important; min-height:100vh !important; }
body { font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif !important; font-size: 14px !important; line-height: 1.55 !important; }
h1 { font-size: 24px !important; margin: 0 0 8px !important; }
h2 { font-size: 16px !important; margin: 12px 0 6px !important; }
.article, .main, article, .content { padding: 12px 14px 22px !important; }
p { margin: 0 0 8px !important; text-align: justify; }
.shell, .layout { min-height: 100vh !important; }
""",
    },
    {
        "name": "magazine_fill",
        "css": """
html, body { margin:0 !important; min-height:100vh !important; }
body { font-family: "Iowan Old Style", Palatino, Georgia, serif !important; font-size: 13.5px !important; line-height: 1.4 !important; }
h1 { font-size: 28px !important; margin: 0 0 6px !important; }
h2 { font-size: 16px !important; margin: 10px 0 4px !important; }
.article, .main, article, .content { padding: 12px 16px 24px !important; }
p { margin: 0 0 7px !important; }
.two { columns: 2 !important; column-gap: 16px !important; }
.shell, .layout { min-height: 100vh !important; }
""",
    },
    {
        "name": "mono_docs",
        "css": """
html, body { margin:0 !important; min-height:100vh !important; }
body { font-family: ui-monospace, "Cascadia Code", Consolas, monospace !important; font-size: 12px !important; line-height: 1.35 !important; }
h1 { font-size: 18px !important; margin: 0 0 6px !important; }
h2 { font-size: 14px !important; margin: 10px 0 4px !important; }
.article, .main, article, .content { padding: 10px 12px 18px !important; }
p { margin: 0 0 5px !important; }
.shell, .layout { min-height: 100vh !important; }
""",
    },
]

# Airier layouts for minority sparse pages
LAYOUT_SPARSE = [
    {
        "name": "serif_roomy",
        "css": """
html, body { margin:0 !important; }
body { font-family: Georgia, serif !important; font-size: 16px !important; line-height: 1.7 !important; background:#f6f4ef !important; }
h1 { font-size: 32px !important; margin: 0 0 16px !important; }
h2 { font-size: 20px !important; margin: 28px 0 12px !important; }
.article, .main, article, .content { padding: 48px 56px 80px !important; max-width: 720px !important; margin: 0 auto !important; }
p { margin: 0 0 18px !important; }
.shell, .layout { min-height: 100vh !important; }
""",
    },
    {
        "name": "slides_air",
        "css": """
html, body { margin:0 !important; }
body { font-family: system-ui, sans-serif !important; font-size: 18px !important; line-height: 1.55 !important; }
h1 { font-size: 36px !important; margin: 0 0 24px !important; }
.article, .main, article, .content, .slide { padding: 64px 72px !important; }
p { margin: 0 0 20px !important; max-width: 36em; }
""",
    },
]

LAYOUT_VARIANTS = LAYOUT_DENSE  # backward-compatible default name

# Target *page* text occupancy bands (union ink / window). Round-robin in Q1.
OCCUPANCY_BANDS: tuple[str, ...] = ("occ_30", "occ_40", "occ_50", "occ_60")
_OCC_PRESETS: dict[str, dict[str, float | int]] = {
    "occ_30": {"font": 16.5, "pad": 40, "extra": 5, "line": 1.65},
    "occ_40": {"font": 15.0, "pad": 24, "extra": 9, "line": 1.50},
    "occ_50": {"font": 14.0, "pad": 14, "extra": 14, "line": 1.40},
    "occ_60": {"font": 13.0, "pad": 8, "extra": 18, "line": 1.32},
}


def occupancy_layout(band: str) -> dict[str, str | int]:
    key = band if band in _OCC_PRESETS else "occ_40"
    p = _OCC_PRESETS[key]
    pad = int(p["pad"])
    font = float(p["font"])
    line = float(p["line"])
    gap = max(4, int(font * 0.4))
    css = f"""
html, body {{ margin:0 !important; min-height:100vh !important; }}
body {{ font-size: {font}px !important; line-height: {line} !important; }}
.article, .main, article, .content, .wrap, .page, .col, .body {{
  padding: {pad}px {int(pad * 1.15)}px !important;
}}
p {{ margin: 0 0 {gap}px !important; }}
"""
    return {"name": key, "css": css, "extra": int(p["extra"])}


def bucket_for_slot(tag: str, bid: str, current_len: int) -> str:
    b = bid.lower()
    t = (tag or "").lower()
    if t in {"h1", "h2", "h3", "h4"} or b.startswith(("h1", "h2", "h3", "title", "side_h")):
        return "headings"
    if t in {"td", "th"} or "cell" in b or (b.startswith(("c", "r")) and current_len < 12):
        return "cells"
    if t in {"code", "pre"} or "code" in b or b.startswith("cmd"):
        return "code_lines"
    if t == "li" or b.startswith(("li", "side_", "nav")):
        return "list_items" if current_len < 80 else "paragraphs"
    if t in {"span", "small"} or "meta" in b or "caption" in b or current_len < 40:
        return "captions"
    return "paragraphs"


def _pair_key(en: str) -> str:
    return " ".join((en or "").casefold().split())


def normalize_pool_item(item: object) -> dict[str, str] | None:
    if isinstance(item, str):
        en = item.strip()
        if not en:
            return None
        return {"en": en, "zh": "", "domain": "legacy", "source": "legacy"}
    if not isinstance(item, dict):
        return None
    en = str(item.get("en") or item.get("src") or item.get("text") or "").strip()
    zh = str(item.get("zh") or item.get("tgt") or item.get("translation") or "").strip()
    if not en:
        return None
    return {
        "en": en,
        "zh": zh,
        "domain": str(item.get("domain") or "unknown"),
        "source": str(item.get("source") or item.get("corpus") or ""),
    }


class UniquePicker:
    """Pick pool strings without repeating the same English text on one page."""

    def __init__(self, pool: dict[str, list], rng: random.Random):
        self.rng = rng
        self.bags: dict[str, list[dict[str, str]]] = {}
        for k, v in (pool or {}).items():
            if not isinstance(v, list):
                continue
            pairs = [p for x in v if (p := normalize_pool_item(x))]
            self.bags[k] = self.rng.sample(pairs, k=len(pairs)) if pairs else []
        self.used: set[str] = set()

    def pick_pair(self, bucket: str) -> dict[str, str]:
        order = [bucket, "paragraphs", "list_items", "captions", "headings", "code_lines", "cells"]
        seen_b: set[str] = set()
        for b in order:
            if b in seen_b:
                continue
            seen_b.add(b)
            bag = self.bags.get(b) or []
            while bag:
                pair = bag.pop()
                key = _pair_key(pair["en"])
                if key and key not in self.used:
                    self.used.add(key)
                    return pair
        n = len(self.used) + 1
        text = f"Unique filler paragraph {n} for synth page diversity."
        self.used.add(_pair_key(text))
        return {"en": text, "zh": f"合成页多样性填充段落 {n}。", "domain": "fallback", "source": "fallback"}

    def pick(self, bucket: str) -> str:
        return self.pick_pair(bucket)["en"]


class PairAllocator:
    """Consume bilingual pairs without replacement across pages (parent process)."""

    def __init__(self, pool: dict[str, list], rng: random.Random):
        self.rng = rng
        self.originals: dict[str, list[dict[str, str]]] = {}
        self.bags: dict[str, list[dict[str, str]]] = {}
        for k, v in (pool or {}).items():
            if not isinstance(v, list):
                continue
            pairs = [p for x in v if (p := normalize_pool_item(x))]
            self.originals[k] = list(pairs)
            self.bags[k] = rng.sample(pairs, k=len(pairs)) if pairs else []
        self.recycled = 0

    def take_page(self, n_per_bucket: int = 12) -> dict[str, list[dict[str, str]]]:
        out: dict[str, list[dict[str, str]]] = {}
        for k, bag in self.bags.items():
            n = min(int(n_per_bucket), max(len(bag), 0))
            if n == 0 and self.originals.get(k):
                # Whole-pool unique pairs already used; reuse across pages.
                refill = list(self.originals[k])
                self.rng.shuffle(refill)
                self.bags[k] = refill
                bag = refill
                n = min(int(n_per_bucket), len(bag))
                self.recycled += n
            out[k] = [bag.pop() for _ in range(n)]
        return out

    def remaining(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.bags.items()}


_SKIP_FILL_KINDS = frozenset({"formula", "image", "chart"})


def _apply_pair(el, soup: BeautifulSoup, pair: dict[str, str]) -> None:
    text = pair["en"]
    tag = el.name or "div"
    el.clear()
    if "\n" in text and tag in {"pre", "code", "p", "div", "li"}:
        for j, line in enumerate(text.split("\n")):
            if j:
                el.append(soup.new_tag("br"))
            el.append(line)
    else:
        el.append(text)
    if pair.get("zh"):
        el["data-mt-zh"] = pair["zh"]
    if pair.get("domain"):
        el["data-mt-domain"] = pair["domain"]


def _should_skip_fill(el) -> bool:
    kind = str(el.get("data-a2-kind") or "").lower()
    if el.get("data-latex") or kind in _SKIP_FILL_KINDS:
        return True
    return False


def _inject_extra_paragraphs(
    soup: BeautifulSoup,
    picker: UniquePicker,
    rng: random.Random,
    n: int,
) -> None:
    """Append extra dense paragraphs into the main article/main container."""
    if n <= 0:
        return
    host = (
        soup.select_one("article.article")
        or soup.select_one(".main")
        or soup.select_one("article")
        or soup.select_one("main")
        or soup.select_one(".shell")
        or soup.select_one(".wrap")
        or soup.body
    )
    if host is None:
        return
    for i in range(n):
        p = soup.new_tag("p")
        p["data-block-id"] = f"extra_p{i}"
        _apply_pair(p, soup, picker.pick_pair("paragraphs"))
        host.append(p)
    if rng.random() < 0.7:
        h = soup.new_tag("h2")
        h["data-block-id"] = "extra_h"
        _apply_pair(h, soup, picker.pick_pair("headings"))
        host.append(h)
        for j in range(2):
            p = soup.new_tag("p")
            p["data-block-id"] = f"extra_h_p{j}"
            _apply_pair(p, soup, picker.pick_pair("paragraphs"))
            host.append(p)


def densify_static_html(
    html: str,
    pool: dict[str, list[str]],
    rng: random.Random,
    *,
    occupancy_band: str,
    extra_paragraphs: int | None = None,
) -> tuple[str, str]:
    """CSS occupancy band + filler paragraphs for authored/static templates."""
    layout = occupancy_layout(occupancy_band)
    n_extra = int(layout["extra"]) if extra_paragraphs is None else int(extra_paragraphs)
    soup = BeautifulSoup(html, "lxml")
    picker = UniquePicker(pool, rng)
    _inject_extra_paragraphs(soup, picker, rng, n_extra)
    style_tag = soup.new_tag("style")
    style_tag.string = f"/* layout:{layout['name']} */\n{layout['css']}"
    if soup.head:
        soup.head.append(style_tag)
    else:
        soup.insert(0, style_tag)
    return str(soup), str(layout["name"])


def inject_occupancy_css(html: str, band: str) -> tuple[str, str]:
    """Paint occupancy CSS onto authored/static HTML without rewriting slots."""
    layout = occupancy_layout(band)
    soup = BeautifulSoup(html, "lxml")
    style_tag = soup.new_tag("style")
    style_tag.string = f"/* layout:{layout['name']} */\n{layout['css']}"
    if soup.head:
        soup.head.append(style_tag)
    else:
        soup.insert(0, style_tag)
    return str(soup), str(layout["name"])


def fill_from_pool(
    template_html: str,
    pool: dict[str, list[str]],
    rng: random.Random,
    *,
    extra_paragraphs: int | None = 6,
    layout_set: str = "dense",
    occupancy_band: str | None = None,
) -> tuple[str, str]:
    soup = BeautifulSoup(template_html, "lxml")
    if occupancy_band:
        layout = occupancy_layout(occupancy_band)
        n_extra = int(layout["extra"]) if extra_paragraphs is None else int(extra_paragraphs)
    else:
        layouts = LAYOUT_SPARSE if layout_set == "sparse" else LAYOUT_DENSE
        if layout_set == "any":
            layouts = LAYOUT_DENSE + LAYOUT_SPARSE
        layout = rng.choice(layouts)
        n_extra = 6 if extra_paragraphs is None else int(extra_paragraphs)
    picker = UniquePicker(pool, rng)

    # Fill existing slots first (unique English within page)
    for el in soup.find_all(attrs={"data-block-id": True}):
        if _should_skip_fill(el):
            continue
        bid = str(el.get("data-block-id"))
        tag = el.name or "div"
        cur = el.get_text(" ", strip=True)
        bucket = bucket_for_slot(tag, bid, len(cur))
        _apply_pair(el, soup, picker.pick_pair(bucket))

    _inject_extra_paragraphs(soup, picker, rng, n_extra)

    style_tag = soup.new_tag("style")
    style_tag.string = f"/* layout:{layout['name']} */\n{layout['css']}"
    if soup.head:
        soup.head.append(style_tag)
    else:
        soup.insert(0, style_tag)
    return str(soup), layout["name"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-paragraphs", type=int, default=8, help="Injected body paras for density")
    ap.add_argument(
        "--layout-set",
        choices=("dense", "sparse", "any"),
        default="dense",
        help="Typography/spacing recipe family",
    )
    ap.add_argument("--pack", type=Path, default=None, help="Optional legacy pack JSON")
    args = ap.parse_args()

    html = args.template.read_text(encoding="utf-8")
    rng = random.Random(args.seed)

    if args.pack is not None:
        pack = json.loads(args.pack.read_text(encoding="utf-8"))
        blocks = pack.get("blocks") or {}
        soup = BeautifulSoup(html, "lxml")
        for el in soup.find_all(attrs={"data-block-id": True}):
            bid = str(el.get("data-block-id"))
            if bid in blocks and blocks[bid] is not None:
                el.clear()
                el.append(str(blocks[bid]))
        layout_name = "pack"
        filled = str(soup)
    else:
        data = json.loads(args.pool.read_text(encoding="utf-8"))
        pool = data.get("pool") or data
        if not isinstance(pool, dict):
            raise SystemExit("pool file must contain a 'pool' object")
        filled, layout_name = fill_from_pool(
            html,
            pool,
            rng,
            extra_paragraphs=args.extra_paragraphs,
            layout_set=args.layout_set,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(filled, encoding="utf-8")
    print(f"wrote {args.out} layout={layout_name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build stratified A1_v2 preview for human review (before full 15k).

Uses MARKER_SPEC (magenta X, 45°) + current POINT_PROMPT.
Covers: long_center, adjacency_light, multi_frag_light, corner_light, empty_neg.

Example:
  export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
  uv run python data/scripts/build_a1_preview_v2.py \\
    --out data/processed/preview_a1_v2 \\
    --gallery data/review_galleries/preview_a1_v2
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from apply_content_pack import fill_from_pool  # noqa: E402
from point_ocr.build_point import BlockAnno, build_point_samples_for_page, load_blocks_json  # noqa: E402
from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.filter_qa import filter_block_label  # noqa: E402
from point_ocr.marker import MARKER_SPEC, draw_crosshair  # noqa: E402
from point_ocr.noise import apply_screen_noise  # noqa: E402
from point_ocr.sample_points import BBox  # noqa: E402
from point_ocr.synth.render import render_html_file_sync  # noqa: E402

DENSE = [
    "01_article_twocol.html",
    "04_zh_news_twocol.html",
    "08_github_readme_dark.html",
    "12_wikipedia_article.html",
    "14_magazine_3col.html",
    "20_forum_zh.html",
    "24_docs_portal_dense.html",
]


def _load_pool(path: Path) -> dict:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pool") or data


def _fill(template: Path, pool: dict, out_html: Path, *, seed: int, extras: int) -> None:
    html_src = template.read_text(encoding="utf-8")
    filled, _ = fill_from_pool(
        html_src, pool, random.Random(seed), extra_paragraphs=extras, layout_set="dense"
    )
    out_html.write_text(filled, encoding="utf-8")


def _large_blocks(
    blocks: list[BlockAnno],
    page_area: float,
    *,
    min_area: float = 0.015,
    min_chars: int = 40,
) -> list[BlockAnno]:
    kept: list[BlockAnno] = []
    for b in blocks:
        fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
        if not fr.keep:
            continue
        if b.ink_area() / page_area < min_area:
            continue
        if len((b.markdown or "").strip()) < min_chars:
            continue
        kept.append(b)
    kept.sort(key=lambda b: b.ink_area(), reverse=True)
    return kept


def _tag(samples: list[PointSample], slice_name: str) -> list[PointSample]:
    for s in samples:
        s.meta["a1_slice"] = slice_name
        s.meta["marker"] = "x45"
    return samples


def _marker_demo(out_dir: Path) -> Path:
    """Same X on light / dark / mid backgrounds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []

    labels = [("light", (245, 245, 240)), ("dark", (28, 30, 36)), ("mid", (120, 130, 140))]
    for name, bg in labels:
        img = Image.new("RGB", (420, 280), bg)
        draw = ImageDraw.Draw(img)
        ink = (30, 30, 30) if name == "light" else ((230, 230, 230) if name == "dark" else (40, 40, 50))
        for i in range(6):
            y = 40 + i * 36
            draw.rectangle([40, y, 380, y + 18], fill=ink)
        marked = draw_crosshair(img, 210, 140, MARKER_SPEC)
        d2 = ImageDraw.Draw(marked)
        d2.text((12, 8), name, fill=(0, 0, 0) if name == "light" else (255, 255, 255))
        panels.append(marked)
    canvas = Image.new("RGB", (420 * 3 + 16, 280), (255, 255, 255))
    for i, p in enumerate(panels):
        canvas.paste(p, (i * (420 + 8), 0))
    path = out_dir / "marker_x45_demo.jpg"
    canvas.save(path, quality=92)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--templates-dir", type=Path, default=ROOT / "data/synth/templates")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/preview_a1_v2")
    ap.add_argument("--gallery", type=Path, default=ROOT / "data/review_galleries/preview_a1_v2")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--skip-gallery", action="store_true")
    args = ap.parse_args()
    use_noise = not args.no_noise

    if not args.pool.is_file():
        raise SystemExit(f"Missing pool {args.pool}")

    pool = _load_pool(args.pool)
    rng = random.Random(args.seed)
    filled_dir = args.out / "filled_html"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"
    demo_dir = args.out / "marker_demo"
    for d in (filled_dir, render_dir, marked_dir):
        d.mkdir(parents=True, exist_ok=True)

    demo_path = _marker_demo(demo_dir)
    print(f"[marker_demo] {demo_path}")

    all_samples: list[PointSample] = []
    slice_counts: Counter[str] = Counter()

    def add(batch: list[PointSample], slice_name: str) -> None:
        _tag(batch, slice_name)
        all_samples.extend(batch)
        slice_counts[slice_name] += len(batch)

    def render_one(tmpl_name: str, page_id: str, seed: int) -> tuple[Image.Image, list[BlockAnno]]:
        tmpl = args.templates_dir / tmpl_name
        filled = filled_dir / f"{page_id}.html"
        _fill(tmpl, pool, filled, seed=seed, extras=6)
        meta = render_html_file_sync(filled, render_dir, page_id=page_id)
        img = Image.open(meta["image_path"]).convert("RGB")
        if use_noise:
            img = apply_screen_noise(img, rng=random.Random(seed + 7))
        blocks = load_blocks_json(Path(meta["blocks_path"]))
        return img, blocks

    # Prefer mid/lower blocks to fight "first block" bias
    def pick_spread(blocks: list[BlockAnno], n: int) -> list[BlockAnno]:
        if len(blocks) <= n:
            return list(blocks)
        blocks = sorted(blocks, key=lambda b: b.bbox.y0)
        # skip top 20% reading-order, take from remaining
        skip = max(1, len(blocks) // 5)
        pool_b = blocks[skip:] or blocks
        rng.shuffle(pool_b)
        # still include one large block
        large = sorted(pool_b, key=lambda b: b.ink_area(), reverse=True)
        chosen = []
        for b in large:
            if b not in chosen:
                chosen.append(b)
            if len(chosen) >= n:
                break
        return chosen

    # ---- long_center ----
    for i, name in enumerate(DENSE[:5]):
        page_id = f"long__{Path(name).stem}__{i}"
        img, blocks = render_one(name, page_id, args.seed + i * 11)
        w, h = img.size
        kept = pick_spread(_large_blocks(blocks, float(w * h)), 3)
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=0,
            seed=args.seed + i,
            avoid_boxes=[r for b in blocks for r in b.ink_rects()],
            coverage="center_band",
            marker="x45",
            meta_extra={"a1_slice": "long_center"},
        )
        pos = [s for s in batch if not s.meta.get("is_negative")][:3]
        add(pos, "long_center")
        print(f"[long_center] {page_id}: +{len(pos)}")

    # ---- adjacency_light (edge_band) ----
    for i, name in enumerate(["01_article_twocol.html", "04_zh_news_twocol.html", "14_magazine_3col.html"]):
        page_id = f"adj__{Path(name).stem}__{i}"
        img, blocks = render_one(name, page_id, args.seed + 100 + i * 13)
        w, h = img.size
        kept = pick_spread(_large_blocks(blocks, float(w * h), min_area=0.01, min_chars=30), 4)
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=0,
            seed=args.seed + 200 + i,
            avoid_boxes=[r for b in blocks for r in b.ink_rects()],
            coverage="edge_band",
            marker="x45",
            meta_extra={"a1_slice": "adjacency_light"},
        )
        pos = [s for s in batch if not s.meta.get("is_negative")][:4]
        add(pos, "adjacency_light")
        print(f"[adjacency_light] {page_id}: +{len(pos)}")

    # ---- multi_frag_light ----
    for i, name in enumerate(["01_article_twocol.html", "14_magazine_3col.html", "04_zh_news_twocol.html"]):
        page_id = f"mfrag__{Path(name).stem}__{i}"
        img, blocks = render_one(name, page_id, args.seed + 300 + i * 17)
        multi = [b for b in blocks if len(b.ink_rects()) > 1 and len((b.markdown or "").strip()) >= 30]
        multi.sort(key=lambda b: b.ink_area(), reverse=True)
        kept = multi[:3] or pick_spread(_large_blocks(blocks, float(img.size[0] * img.size[1])), 2)
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=0,
            seed=args.seed + 310 + i,
            avoid_boxes=[r for b in blocks for r in b.ink_rects()],
            coverage="center_band",
            marker="x45",
            meta_extra={"a1_slice": "multi_frag_light"},
        )
        pos = [s for s in batch if not s.meta.get("is_negative")][:3]
        add(pos, "multi_frag_light")
        print(f"[multi_frag_light] {page_id}: +{len(pos)} (multi={len(multi)})")

    # ---- corner_light ----
    for i, name in enumerate(["12_wikipedia_article.html", "20_forum_zh.html", "24_docs_portal_dense.html"]):
        page_id = f"corner__{Path(name).stem}__{i}"
        img, blocks = render_one(name, page_id, args.seed + 500 + i * 19)
        w, h = img.size
        kept = pick_spread(_large_blocks(blocks, float(w * h)), 3)
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=0,
            seed=args.seed + 520 + i,
            avoid_boxes=[r for b in blocks for r in b.ink_rects()],
            coverage="corner_extreme",
            marker="x45",
            meta_extra={"a1_slice": "corner_light"},
        )
        pos = [s for s in batch if not s.meta.get("is_negative")][:3]
        add(pos, "corner_light")
        print(f"[corner_light] {page_id}: +{len(pos)}")

    # ---- empty_neg ----
    for i, name in enumerate(["08_github_readme_dark.html", "01_article_twocol.html"]):
        page_id = f"empty__{Path(name).stem}__{i}"
        img, blocks = render_one(name, page_id, args.seed + 700 + i)
        avoid = [r for b in blocks for r in b.ink_rects()]
        batch = build_point_samples_for_page(
            img,
            [],
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=5,
            seed=args.seed + 720 + i,
            avoid_boxes=avoid,
            coverage="center_band",
            marker="x45",
            meta_extra={"a1_slice": "empty_neg"},
        )
        neg = [s for s in batch if s.meta.get("is_negative")][:5]
        add(neg, "empty_neg")
        print(f"[empty_neg] {page_id}: -{len(neg)}")

    rng.shuffle(all_samples)
    jsonl = args.out / "point_sharegpt.jsonl"
    n = write_jsonl(jsonl, all_samples)
    print(f"\nWrote {n} → {jsonl}")
    print("Slice counts:", dict(sorted(slice_counts.items())))
    print(f"Open marker demo: {demo_path}")

    if not args.skip_gallery and n > 0:
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "data/scripts/review_sample_gallery.py"),
                "--jsonl",
                str(jsonl),
                "--out",
                str(args.gallery),
                "--n",
                str(min(n, 100)),
                "--seed",
                str(args.seed),
                "--show-bbox",
                "--stratify",
                "a1_slice",
            ]
        )
        print(f"Open gallery: {args.gallery / 'index.html'}")


if __name__ == "__main__":
    main()

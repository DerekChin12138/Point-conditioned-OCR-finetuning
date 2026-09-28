#!/usr/bin/env python3
"""Build a small A1-style preview batch (dozens of samples) for human review.

Mixes dense (majority) and sparse (minority) pages. Then writes jsonl + review gallery
with side-by-side image / GT text.

Example:
  uv run python data/scripts/build_preview_batch.py \\
    --pool data/synth/content_pools/a1_pool.json \\
    --n-pages 10 --sparse-page-frac 0.2 \\
    --gallery data/review_galleries/preview_a1_mix
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from point_ocr.build_point import build_point_samples_for_page, load_blocks_json
from point_ocr.dataset_format import write_jsonl
from point_ocr.filter_qa import filter_block_label
from point_ocr.noise import apply_screen_noise
from point_ocr.synth.render import render_html_file_sync

DENSE_TEMPLATES = [
    "01_article_twocol.html",
    "04_zh_news_twocol.html",
    "08_github_readme_dark.html",
    "12_wikipedia_article.html",
    "14_magazine_3col.html",
    "20_forum_zh.html",
    "24_docs_portal_dense.html",
]
SPARSE_TEMPLATES = [
    "05_pdf_academic.html",
    "15_slides_dark.html",
    "11_email_client.html",
]


def _fill_template(
    template: Path,
    pool_path: Path,
    out_html: Path,
    seed: int,
    extra_paragraphs: int,
    *,
    layout_set: str,
) -> None:
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "data/scripts/apply_content_pack.py"),
            "--template",
            str(template),
            "--pool",
            str(pool_path),
            "--out",
            str(out_html),
            "--seed",
            str(seed),
            "--extra-paragraphs",
            str(extra_paragraphs),
            "--layout-set",
            layout_set,
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--templates-dir", type=Path, default=ROOT / "data/synth/templates")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/preview_a1")
    ap.add_argument("--gallery", type=Path, default=ROOT / "data/review_galleries/preview_a1")
    ap.add_argument("--n-pages", type=int, default=10)
    ap.add_argument("--points-per-page", type=int, default=4)
    ap.add_argument("--negatives", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area-ratio", type=float, default=0.02)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--extra-paragraphs", type=int, default=6)
    ap.add_argument("--sparse-extra-paragraphs", type=int, default=1)
    ap.add_argument(
        "--sparse-page-frac",
        type=float,
        default=0.2,
        help="Share of pages that use sparse templates/layouts",
    )
    ap.add_argument("--noise", action="store_true", default=True)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--skip-gallery", action="store_true")
    args = ap.parse_args()
    use_noise = args.noise and not args.no_noise

    if not args.pool.is_file():
        raise SystemExit(f"Missing pool {args.pool}; run generate_template_content.py first")

    rng = random.Random(args.seed)
    filled_dir = args.out / "filled_html"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"
    for d in (filled_dir, render_dir, marked_dir):
        d.mkdir(parents=True, exist_ok=True)

    dense = [args.templates_dir / n for n in DENSE_TEMPLATES if (args.templates_dir / n).is_file()]
    sparse = [args.templates_dir / n for n in SPARSE_TEMPLATES if (args.templates_dir / n).is_file()]
    if not dense:
        raise SystemExit("No dense templates found")
    if not sparse:
        print("[warn] no sparse templates; all pages dense")

    all_samples = []
    n_sparse_pages = 0
    for i in range(args.n_pages):
        use_sparse = bool(sparse) and (rng.random() < args.sparse_page_frac)
        if use_sparse:
            tmpl = rng.choice(sparse)
            extras = args.sparse_extra_paragraphs
            layout_set = "sparse"
            n_sparse_pages += 1
        else:
            tmpl = dense[i % len(dense)]
            extras = args.extra_paragraphs
            layout_set = "dense"

        page_id = f"{tmpl.stem}__{'sp' if use_sparse else 'dn'}{i}"
        filled = filled_dir / f"{page_id}.html"
        _fill_template(
            tmpl,
            args.pool,
            filled,
            seed=args.seed + i * 17,
            extra_paragraphs=extras,
            layout_set=layout_set,
        )
        meta = render_html_file_sync(filled, render_dir, page_id=page_id)
        img = Image.open(meta["image_path"]).convert("RGB")
        if use_noise:
            img = apply_screen_noise(img)
        w, h = img.size
        page_area = float(w * h)

        min_area = args.min_area_ratio * (0.6 if use_sparse else 1.0)
        min_chars = max(20, args.min_chars - (20 if use_sparse else 0))

        blocks = load_blocks_json(Path(meta["blocks_path"]))
        # Positives: large blocks only. Negatives must avoid ALL on-page text boxes.
        all_boxes = [b.bbox for b in blocks]
        kept = []
        for b in blocks:
            fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
            if not fr.keep:
                continue
            if b.bbox.area() / page_area < min_area:
                continue
            if len(b.markdown.strip()) < min_chars:
                continue
            kept.append(b)
        kept.sort(key=lambda b: b.bbox.area(), reverse=True)
        kept = kept[: max(3, args.points_per_page)]

        if not kept:
            print(f"[warn] no large blocks on {page_id}; skipping positives")
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=2,
            n_negatives=args.negatives,
            seed=args.seed + i,
            avoid_boxes=all_boxes,
        )
        pos = [s for s in batch if not s.meta.get("is_negative")]
        neg = [s for s in batch if s.meta.get("is_negative")]
        for s in pos + neg:
            s.meta["density"] = "sparse" if use_sparse else "dense"
        rng.shuffle(pos)
        pos = pos[: args.points_per_page]
        all_samples.extend(pos + neg)
        print(
            f"[ok] {page_id}: density={'sparse' if use_sparse else 'dense'} "
            f"blocks={len(kept)} samples={len(pos) + len(neg)}"
        )

    jsonl = args.out / "point_sharegpt.jsonl"
    n = write_jsonl(jsonl, all_samples)
    print(f"Wrote {n} → {jsonl} (sparse_pages={n_sparse_pages}/{args.n_pages})")

    if not args.skip_gallery and n > 0:
        g_n = min(n, 48)
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "data/scripts/review_sample_gallery.py"),
                "--jsonl",
                str(jsonl),
                "--out",
                str(args.gallery),
                "--n",
                str(g_n),
                "--seed",
                str(args.seed),
                "--show-bbox",
            ]
        )
        print(f"Open gallery: {args.gallery / 'index.html'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize a small held-out eval set from templates + optional real images.

Categories to cover (manual curation checklist):
  web article, PDF-like, two-column, table cell, formula, code, UI chrome, fullscreen capture
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from point_ocr.build_point import build_point_samples_for_page, load_blocks_json
from point_ocr.dataset_format import write_jsonl
from point_ocr.synth.render import render_html_file_sync


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", type=Path, default=ROOT / "data" / "synth" / "templates")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "heldout")
    ap.add_argument("--seed", type=int, default=123)
    # Fixed small R for eval reproducibility
    ap.add_argument("--r-min", type=int, default=2)
    ap.add_argument("--r-max", type=int, default=2)
    ap.add_argument("--negatives", type=int, default=3)
    args = ap.parse_args()

    render_dir = args.out / "renders"
    marked = args.out / "images"
    all_samples = []
    for i, html in enumerate(sorted(args.templates.glob("*.html"))):
        meta = render_html_file_sync(html, render_dir, page_id=f"hold_{html.stem}")
        img = Image.open(meta["image_path"]).convert("RGB")
        blocks = load_blocks_json(Path(meta["blocks_path"]))
        samples = build_point_samples_for_page(
            img,
            blocks,
            page_id=f"hold_{html.stem}",
            out_image_dir=marked,
            r_min=args.r_min,
            r_max=args.r_max,
            n_negatives=args.negatives,
            seed=args.seed + i,
        )
        all_samples.extend(samples)

    jsonl = args.out / "heldout_sharegpt.jsonl"
    write_jsonl(jsonl, all_samples)
    manifest = [
        {
            "sample_id": s.sample_id,
            "image_path": s.image_path,
            "target": s.target,
            "is_negative": bool(s.meta.get("is_negative")),
            "task": s.task,
        }
        for s in all_samples
    ]
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"held-out samples={len(all_samples)} → {args.out}")


if __name__ == "__main__":
    main()

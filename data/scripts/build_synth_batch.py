#!/usr/bin/env python3
"""Build synthetic POINT dataset from HTML templates (Playwright)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from point_ocr.build_point import build_point_samples_for_page, load_blocks_json
from point_ocr.dataset_format import write_jsonl
from point_ocr.filter_qa import filter_block_label
from point_ocr.noise import apply_screen_noise
from point_ocr.synth.render import render_html_file_sync


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--templates",
        type=Path,
        default=ROOT / "data" / "synth" / "templates",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "synth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-min", type=int, default=3)
    ap.add_argument("--r-max", type=int, default=5)
    ap.add_argument("--negatives", type=int, default=4)
    ap.add_argument("--noise", action="store_true", help="Apply screen-domain noise before marking")
    args = ap.parse_args()

    render_dir = args.out / "renders"
    images_dir = args.out / "marked"
    samples_path = args.out / "point_sharegpt.jsonl"

    all_samples = []
    dropped = 0
    for i, html in enumerate(sorted(args.templates.glob("*.html"))):
        meta = render_html_file_sync(html, render_dir, page_id=html.stem)
        img = Image.open(meta["image_path"]).convert("RGB")
        if args.noise:
            img = apply_screen_noise(img)

        blocks = load_blocks_json(Path(meta["blocks_path"]))
        kept = []
        for b in blocks:
            fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
            if fr.keep:
                kept.append(b)
            else:
                dropped += 1

        page_samples = build_point_samples_for_page(
            img,
            kept,
            page_id=html.stem,
            out_image_dir=images_dir,
            r_min=args.r_min,
            r_max=args.r_max,
            n_negatives=args.negatives,
            seed=args.seed + i,
        )
        all_samples.extend(page_samples)
        print(f"[ok] {html.name}: kept={len(kept)} → samples={len(page_samples)}")

    n = write_jsonl(samples_path, all_samples)
    print(f"Wrote {n} samples (dropped_blocks={dropped}) → {samples_path}")


if __name__ == "__main__":
    main()

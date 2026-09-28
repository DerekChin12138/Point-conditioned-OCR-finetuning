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


# Desktop / multi-window shells — keep HTML templates, skip in Stage A doc-only builds.
DESKTOP_SHELL_PREFIXES = (
    "21_desktop_",
    "22_desktop_",
    "23_ide_",
    "25_desktop_",
    "26_desktop_",
)


def _is_desktop_shell(stem: str) -> bool:
    return any(stem.startswith(p) for p in DESKTOP_SHELL_PREFIXES)


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
    ap.add_argument(
        "--include-desktop",
        action="store_true",
        help="Include desktop/IDE shell templates (default: skip; templates stay on disk)",
    )
    args = ap.parse_args()

    render_dir = args.out / "renders"
    images_dir = args.out / "marked"
    samples_path = args.out / "point_sharegpt.jsonl"

    all_samples = []
    dropped = 0
    skipped_desktop = 0
    for i, html in enumerate(sorted(args.templates.glob("*.html"))):
        if not args.include_desktop and _is_desktop_shell(html.stem):
            skipped_desktop += 1
            print(f"[skip-desktop] {html.name}")
            continue
        meta = render_html_file_sync(html, render_dir, page_id=html.stem)
        img = Image.open(meta["image_path"]).convert("RGB")
        if args.noise:
            img = apply_screen_noise(img)

        blocks = load_blocks_json(Path(meta["blocks_path"]))
        all_boxes = [b.bbox for b in blocks]
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
            avoid_boxes=all_boxes,
        )
        all_samples.extend(page_samples)
        print(f"[ok] {html.name}: kept={len(kept)} → samples={len(page_samples)}")

    n = write_jsonl(samples_path, all_samples)
    print(
        f"Wrote {n} samples (dropped_blocks={dropped}, skipped_desktop={skipped_desktop}) → {samples_path}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render selected synth templates + weak-ring markers + print coord prompts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont

from point_ocr.build_point import load_blocks_json
from point_ocr.marker import MARKER_SPEC, draw_crosshair
from point_ocr.prompts import format_point_prompt
from point_ocr.synth.render import render_html_file_sync


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--templates",
        nargs="*",
        default=[
            "25_desktop_win11_collage.html",
            "26_desktop_mac_collage.html",
        ],
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "preview_samples" / "desktop_collage",
    )
    args = ap.parse_args()

    tmpl_dir = ROOT / "data" / "synth" / "templates"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"
    args.out.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    marked_dir.mkdir(parents=True, exist_ok=True)

    print("MARKER_SPEC:", MARKER_SPEC)

    digit = Image.new("RGB", (400, 160), (245, 245, 245))
    d = ImageDraw.Draw(digit)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    d.text((150, 55), "128", fill=(20, 20, 20), font=font)
    bb = d.textbbox((150, 55), "128", font=font)
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    digit_marked = draw_crosshair(digit, cx, cy)
    digit_path = marked_dir / "marker_coord_ring_digit128.png"
    digit_marked.save(digit_path)
    print("[digit]", digit_path.name)
    print(" ", format_point_prompt(cx, cy, *digit.size))

    for name in args.templates:
        html = tmpl_dir / name
        if not html.exists():
            print(f"[skip] missing {html}")
            continue
        meta = render_html_file_sync(html, render_dir, page_id=html.stem)
        img = Image.open(meta["image_path"]).convert("RGB")
        w, h = img.size
        clean = marked_dir / f"{html.stem}__desktop.png"
        img.save(clean)
        blocks = load_blocks_json(Path(meta["blocks_path"]))
        blocks_sorted = sorted(blocks, key=lambda b: len(b.markdown or ""))
        picks = blocks_sorted[:2] + blocks_sorted[-1:]
        seen: set[str] = set()
        for i, b in enumerate(picks):
            if b.block_id in seen:
                continue
            seen.add(b.block_id)
            x0, y0, x1, y1 = b.bbox.as_tuple()
            px, py = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            marked = draw_crosshair(img, px, py)
            out = marked_dir / f"{html.stem}__ring_{i}_{b.block_id}.png"
            marked.save(out)
            prompt = format_point_prompt(px, py, w, h)
            print(f"[ok] {out.name}  gt={b.markdown[:50]!r}")
            print(f"     {prompt[:120]}...")
        print(f"  size={img.size} n_blocks={len(blocks)} clean={clean.name}")


if __name__ == "__main__":
    main()

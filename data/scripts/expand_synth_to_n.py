#!/usr/bin/env python3
"""Expand synth POINT pairs to a target count using existing renders (+ noise).

Does NOT call any LLM. Uses Playwright renders already on disk (or re-renders
templates if needed), then multiplies via corner sampling + screen noise + seeds.

Example:
  uv run python data/scripts/expand_synth_to_n.py --target 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from point_ocr.build_point import build_point_samples_for_page, load_blocks_json
from point_ocr.dataset_format import write_jsonl
from point_ocr.filter_qa import filter_block_label
from point_ocr.noise import apply_screen_noise


# Desktop / multi-window shells — excluded by default (templates stay on disk).
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
    ap.add_argument("--renders", type=Path, default=ROOT / "data/processed/synth/renders")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/synth")
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--r-min", type=int, default=4)
    ap.add_argument("--r-max", type=int, default=6)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", action="store_true", default=True)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument(
        "--include-desktop",
        action="store_true",
        help="Include desktop/IDE shell renders (default: doc-only)",
    )
    args = ap.parse_args()
    use_noise = args.noise and not args.no_noise

    pages = sorted(args.renders.glob("*.png"))
    if not pages:
        raise SystemExit(f"No renders in {args.renders}; run build_synth_batch.py first")
    if not args.include_desktop:
        pages = [p for p in pages if not _is_desktop_shell(p.stem)]
    if not pages:
        raise SystemExit("No non-desktop renders left; check --include-desktop or rebuild")

    marked = args.out / "marked"
    marked.mkdir(parents=True, exist_ok=True)
    all_samples = []
    round_id = 0

    while len(all_samples) < args.target:
        for png in pages:
            if len(all_samples) >= args.target:
                break
            blocks_path = png.with_suffix(".blocks.json")
            if not blocks_path.exists():
                print(f"[skip] missing blocks for {png.name}")
                continue
            img = Image.open(png).convert("RGB")
            if use_noise:
                img = apply_screen_noise(img)
            blocks_raw = load_blocks_json(blocks_path)
            all_boxes = [b.bbox for b in blocks_raw]
            blocks = []
            for b in blocks_raw:
                fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
                if fr.keep:
                    blocks.append(b)
            page_id = f"{png.stem}__r{round_id}"
            batch = build_point_samples_for_page(
                img,
                blocks,
                page_id=page_id,
                out_image_dir=marked,
                r_min=args.r_min,
                r_max=args.r_max,
                n_negatives=args.negatives,
                seed=args.seed + round_id * 10007 + hash(png.stem) % 997,
                avoid_boxes=all_boxes,
            )
            all_samples.extend(batch)
            print(f"[ok] {page_id}: +{len(batch)} → total {len(all_samples)}")
        round_id += 1
        if round_id > 200:
            raise SystemExit("Too many rounds; check renders/blocks")

    # Trim to exact target (keep prefix for reproducibility)
    all_samples = all_samples[: args.target]
    out_jsonl = args.out / "point_sharegpt.jsonl"
    n = write_jsonl(out_jsonl, all_samples)
    meta = {
        "protocol": "thick_crosshair_static_prompt",
        "scope": "doc_only" if not args.include_desktop else "all_templates",
        "target": args.target,
        "n_written": n,
        "rounds": round_id,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "negatives": args.negatives,
        "noise": use_noise,
        "include_desktop": args.include_desktop,
        "n_source_pages": len(pages),
    }
    (args.out / "expand_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote {n} → {out_jsonl}")


if __name__ == "__main__":
    main()

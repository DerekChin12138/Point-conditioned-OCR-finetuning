#!/usr/bin/env python3
"""Rewrite Q1 ShareGPT splits → q1_coordinate.

Same diamond5 5×2 points as Q1, but images are unmarked render PNGs
and the prompt uses pixel point_2d. Does not rebuild pools.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.coord_condition import coord_sharegpt_row, render_path_for_meta  # noqa: E402


def _assistant_text(row: dict) -> str:
    for m in row.get("messages") or []:
        if m.get("role") == "assistant":
            return str(m.get("content") or "")
    return ""


def rewrite_split(src: Path, dst: Path, *, pools_root: Path) -> dict:
    from PIL import Image

    n_ok = 0
    n_skip = 0
    n_size = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = dict(row.get("metadata") or {})
            point = meta.get("point") or []
            imgs = row.get("images") or []
            if len(point) < 2:
                n_skip += 1
                continue
            try:
                render = render_path_for_meta(meta, pools_root=pools_root)
            except ValueError:
                n_skip += 1
                continue
            if not render.is_file():
                n_skip += 1
                continue
            marked = Path(str(imgs[0])) if imgs else None
            with Image.open(render) as im:
                image_w, image_h = im.size
            if marked is not None and marked.is_file():
                with Image.open(marked) as im:
                    mw, mh = im.size
                # Point was sampled on the Q1 marked raster. Skip if the current
                # unmarked PNG was re-rendered at a different size.
                if (mw, mh) != (image_w, image_h):
                    n_size += 1
                    n_skip += 1
                    continue
            out = coord_sharegpt_row(
                sample_id=str(meta.get("sample_id") or f"coord_{n_ok}"),
                image_path=str(render.resolve()),
                target=_assistant_text(row),
                px=float(point[0]),
                py=float(point[1]),
                image_w=image_w,
                image_h=image_h,
                meta=meta,
            )
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_ok += 1
    return {
        "src": str(src),
        "dst": str(dst),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_size_mismatch": n_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=ROOT / "data/splits_stage_q1")
    ap.add_argument("--dst-dir", type=Path, default=ROOT / "data/splits_stage_q1_coordinate")
    ap.add_argument("--pools-root", type=Path, default=ROOT / "data/pools_q")
    args = ap.parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        src = args.src_dir / name
        if not src.is_file():
            print(f"skip missing {src}", flush=True)
            continue
        stats = rewrite_split(src, args.dst_dir / name, pools_root=args.pools_root)
        summary.append(stats)
        print(
            f"{name}: ok={stats['n_ok']} skip={stats['n_skip']} "
            f"size_mismatch={stats['n_size_mismatch']}",
            flush=True,
        )
    (args.dst_dir / "rewrite_meta.json").write_text(
        json.dumps(
            {
                "from": str(args.src_dir),
                "prompt_key": "coord_q1",
                "coord_origin": "top-left",
                "coord_unit": "pixel",
                "marker": "none",
                "note": "Same Q1 diamond5 points; unmarked renders whose pixel size matches the Q1 marked JPEG; pixel point_2d. No visual X.",
                "splits": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

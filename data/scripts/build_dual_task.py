#!/usr/bin/env python3
"""Build Stage B dual-task JSON: POINT samples + PAGE samples (unmarked pages).

PAGE samples use original page images (no crosshair) + PAGE_PROMPT + full-page markdown.
POINT samples come from existing point_sharegpt.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import load_jsonl
from point_ocr.prompts import PAGE_PROMPT, POINT_PROMPT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-jsonl", type=Path, required=True)
    ap.add_argument(
        "--page-manifest",
        type=Path,
        required=True,
        help='JSON list of {"image_path","markdown","page_id"} for full-page PAGE task',
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--page-ratio", type=float, default=0.3, help="Approx fraction PAGE among dual set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    point = load_jsonl(args.point_jsonl)
    pages = json.loads(args.page_manifest.read_text(encoding="utf-8"))

    page_recs = []
    for p in pages:
        page_recs.append(
            {
                "messages": [
                    {"role": "user", "content": f"<image>{PAGE_PROMPT}"},
                    {"role": "assistant", "content": p["markdown"]},
                ],
                "images": [p["image_path"]],
                "metadata": {"task": "PAGE", "page_id": p.get("page_id")},
            }
        )

    # Target mix
    n_page = int(round(len(point) * args.page_ratio / max(1e-6, 1 - args.page_ratio)))
    if len(page_recs) >= n_page:
        page_use = rng.sample(page_recs, n_page)
    else:
        page_use = list(page_recs)
        while page_recs and len(page_use) < n_page:
            page_use.append(rng.choice(page_recs))

    # Ensure POINT prompts still match contract (rewrite if needed)
    for rec in point:
        msgs = rec.get("messages") or []
        if msgs and isinstance(msgs[0].get("content"), str):
            content = msgs[0]["content"]
            if content.startswith("<image>"):
                msgs[0]["content"] = f"<image>{POINT_PROMPT}"

    mixed = point + page_use
    rng.shuffle(mixed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mixed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dual samples={len(mixed)} (point={len(point)}, page={len(page_use)}) → {args.out}")


if __name__ == "__main__":
    main()

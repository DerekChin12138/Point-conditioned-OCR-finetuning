#!/usr/bin/env python3
"""Compose Q1-withreal splits.

Synthetic: Q1 absolute counts, multi_frag 3750 (was 1875). Magenta X + a2_v3.
Real: all of data/pools_real/real_labeled, split 90/5/5, appended.

multi_frag pool only has 2800 rows. Before sampling, stamp diamond5 on the
*other* fragment of each already-labeled block (same page, same target) so
the pool can supply 3750 without new HTML renders.

  uv run python data/scripts/compose_q1_withreal.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import load_jsonl  # noqa: E402
from point_ocr.marker import (  # noqa: E402
    CURRENT_MARKER_TAG,
    draw_crosshair,
    sample_marker_area_frac,
    spec_for_image,
)  # noqa: E402
from point_ocr.pools.compose import (  # noqa: E402
    interleave_stage,
    load_recipe,
    mix_summary,
    sample_from_pools,
    split_counts,
    split_rows,
    write_rows,
)
from point_ocr.pools.spec import quota_from_frac  # noqa: E402
from point_ocr.prompts import pixel_to_norm  # noqa: E402
from point_ocr.sample_points import BBox, sample_diamond5_points  # noqa: E402

RECIPE = ROOT / "data/recipes/stage_q1_withreal.yaml"
POOLS_Q = ROOT / "data/pools_q"
REAL_JSONL = ROOT / "data/pools_real/real_labeled/point_sharegpt.jsonl"
OUT = ROOT / "data/splits_stage_q1_withreal"
MULTI = POOLS_Q / "multi_frag"


def _expand_multi_frag_alt() -> int:
    """Stamp the unused fragment of each multi_frag block. Idempotent."""
    jsonl = MULTI / "point_sharegpt.jsonl"
    rows = load_jsonl(jsonl)
    by_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("metadata") or {}
        by_block[(str(meta.get("page_id")), str(meta.get("block_id")))].append(row)

    added: list[dict[str, Any]] = []
    marked_dir = MULTI / "marked"
    for (page_id, block_id), members in by_block.items():
        used = {int(m["metadata"].get("fragment_index") or 0) for m in members}
        bboxes = members[0]["metadata"].get("bboxes") or []
        if len(bboxes) < 2:
            continue
        tmpl = members[0]
        img_path = MULTI / "renders" / f"{page_id}.png"
        if not img_path.is_file():
            continue
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        for fi, bb in enumerate(bboxes):
            if fi in used:
                continue
            sid_prefix = f"{page_id}:{block_id}:altf{fi}"
            if any(str((r.get("metadata") or {}).get("sample_id", "")).startswith(sid_prefix) for r in rows):
                continue
            box = BBox(*map(float, bb))
            if box.width() < 4 or box.height() < 4:
                continue
            digest = hashlib.md5(f"{page_id}|{block_id}|{fi}|alt".encode()).hexdigest()
            rng = random.Random(int(digest[:8], 16))
            pts = sample_diamond5_points(
                box,
                rng=rng,
                block_id=block_id,
                image_w=w,
                image_h=h,
                fragment_index=fi,
            )
            for j, pt in enumerate(pts):
                frac = sample_marker_area_frac(rng)
                marked = draw_crosshair(img, pt.x, pt.y, spec_for_image(w, h, area_frac=frac))
                name = f"{page_id}__{block_id}__altf{fi}_p{j}.jpg"
                path = marked_dir / name
                marked.save(path, format="JPEG", quality=92)
                nx, ny = pixel_to_norm(pt.x, pt.y, w, h)
                meta = dict(tmpl.get("metadata") or {})
                meta.update(
                    {
                        "sample_id": f"{sid_prefix}_{j}",
                        "point": [pt.x, pt.y],
                        "point_norm": [nx, ny],
                        "region": pt.region,
                        "bbox": list(box.as_tuple()),
                        "fragment_index": fi,
                        "alt_fragment": True,
                        "marker": CURRENT_MARKER_TAG,
                        "marker_area_frac": float(frac),
                    }
                )
                added.append(
                    {
                        "messages": tmpl.get("messages"),
                        "images": [str(path.resolve())],
                        "metadata": meta,
                    }
                )
        img.close()

    if not added:
        print(f"[multi_frag] no new alt-fragment rows (have {len(rows)})", flush=True)
        return len(rows)
    out_rows = rows + added
    write_rows(jsonl, out_rows)
    print(f"[multi_frag] {len(rows)} + {len(added)} alt-fragment → {len(out_rows)}", flush=True)
    return len(out_rows)


def _append_real(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    real = load_jsonl(REAL_JSONL)
    for row in real:
        meta = dict(row.get("metadata") or {})
        meta["pool_id"] = "real_labeled"
        meta["stage"] = "stage_q1_withreal"
        meta["prompt_key"] = "a2_v3"
        row["metadata"] = meta
    rng.shuffle(real)
    n_train, n_val, n_test = split_counts(len(real), 0.90, 0.05)
    r_train = real[:n_train]
    r_val = real[n_train : n_train + n_val]
    r_test = real[n_train + n_val : n_train + n_val + n_test]
    return train + r_train, val + r_val, test + r_test, {
        "n_real": len(real),
        "real_train": len(r_train),
        "real_val": len(r_val),
        "real_test": len(r_test),
    }


def main() -> None:
    _expand_multi_frag_alt()
    recipe = load_recipe(RECIPE)
    n = int(recipe["n"])
    mix = recipe["mix"]
    quotas = quota_from_frac(mix, n)
    seed = int(recipe.get("seed") or 42)
    rng = random.Random(seed)
    print(f"[compose] quotas={quotas}", flush=True)
    rows = sample_from_pools(
        mix,
        n,
        POOLS_Q,
        rng=rng,
        prompt_key="a2_v3",
        chrome_frac=float(recipe["chrome_frac"]),
    )
    for row in rows:
        meta = dict(row.get("metadata") or {})
        meta["stage"] = "stage_q1_withreal"
        row["metadata"] = meta
    train, val, test = split_rows(rows, train_frac=0.90, val_frac=0.05, rng=rng)
    train, val, test, real_stats = _append_real(train, val, test, rng)
    train = interleave_stage(train, rng)
    val = interleave_stage(val, rng)
    test = interleave_stage(test, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    n_train = write_rows(OUT / "train.jsonl", train)
    n_val = write_rows(OUT / "val.jsonl", val)
    n_test = write_rows(OUT / "test.jsonl", test)
    meta = {
        "name": "stage_q1_withreal",
        "prompt_key": "a2_v3",
        "marker": "x45c",
        "synthetic_n": n,
        "synthetic_quotas": quotas,
        **real_stats,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train": mix_summary(train),
        "val": mix_summary(val),
        "test": mix_summary(test),
    }
    (OUT / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n_train", "n_val", "n_test", "synthetic_quotas", "real_train", "real_val", "real_test", "train")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

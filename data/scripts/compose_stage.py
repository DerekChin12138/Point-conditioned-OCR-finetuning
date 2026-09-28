#!/usr/bin/env python3
"""Compose train/val/test JSONL from objective pools using a stage recipe.

Does not re-render. Image paths stay inside data/pools/<id>/marked/.

  uv run python data/scripts/compose_stage.py --recipe data/recipes/stage_a1.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.pools.compose import (  # noqa: E402
    interleave_stage,
    load_recipe,
    mix_summary,
    sample_from_pools,
    split_rows,
    write_rows,
)
from point_ocr.pools.spec import quota_from_frac  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--pools-root", type=Path, default=ROOT / "data/pools")
    ap.add_argument("--out", type=Path, default=None, help="Default: data/splits_<recipe.name>")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    recipe = load_recipe(args.recipe)
    name = str(recipe["name"])
    n = int(recipe["n"])
    mix = recipe["mix"]
    split = recipe["split"]
    stem_boost = recipe.get("stem_boost") or {}
    chrome_frac = recipe.get("chrome_frac")
    seed = int(args.seed if args.seed is not None else recipe.get("seed") or 42)
    prompt_key = recipe.get("prompt_key")
    out = args.out or (ROOT / "data" / f"splits_{name}")
    rng = random.Random(seed)

    quotas = quota_from_frac(mix, n)
    print(f"[compose] {name} n={n} quotas={quotas} chrome_frac={chrome_frac} out={out}", flush=True)
    if stem_boost:
        print(f"[compose] stem_boost={stem_boost}", flush=True)
    rows = sample_from_pools(
        mix,
        n,
        args.pools_root,
        rng=rng,
        prompt_key=str(prompt_key) if prompt_key else None,
        stem_boost=stem_boost,
        chrome_frac=float(chrome_frac) if chrome_frac is not None else None,
    )
    for row in rows:
        meta = row.get("metadata") or {}
        meta["stage"] = name
        row["metadata"] = meta

    train_frac = float(split.get("train", 0.90))
    val_frac = float(split.get("val", 0.05))
    train, val, test = split_rows(rows, train_frac=train_frac, val_frac=val_frac, rng=rng)
    train = interleave_stage(train, rng)
    val = interleave_stage(val, rng)
    test = interleave_stage(test, rng)

    out.mkdir(parents=True, exist_ok=True)
    n_train = write_rows(out / "train.jsonl", train)
    n_val = write_rows(out / "val.jsonl", val)
    n_test = write_rows(out / "test.jsonl", test)
    meta = {
        "recipe": str(args.recipe),
        "name": name,
        "n": n,
        "mix": mix,
        "chrome_frac": chrome_frac,
        "quotas": quotas,
        "prompt_key": prompt_key,
        "seed": seed,
        "pools_root": str(args.pools_root),
        "split_frac": split,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train": mix_summary(train),
        "val": mix_summary(val),
        "test": mix_summary(test),
    }
    (out / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[split] {name} train={n_train} val={n_val} test={n_test} "
        f"train_pools={meta['train']['by_pool']}",
        flush=True,
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

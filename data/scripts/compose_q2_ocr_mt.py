#!/usr/bin/env python3
"""Compose Q2 OCR-MT splits (2× Q1 mix, ocr_mt_v1 prompt).

Synthetic from data/pools_q2. Real from data/pools_real/real_labeled_mt
if present, else real_labeled (OCR-only; wrap later).

  uv run python data/scripts/compose_q2_ocr_mt.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import load_jsonl  # noqa: E402
from point_ocr.ocr_mt import wrap_point_target  # noqa: E402
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

RECIPE = ROOT / "data/recipes/stage_q2_ocr_mt.yaml"
POOLS_Q2 = ROOT / "data/pools_q2"
REAL_MT = ROOT / "data/pools_real/real_labeled_mt/point_sharegpt.jsonl"
REAL_OCR = ROOT / "data/pools_real/real_labeled/point_sharegpt.jsonl"
OUT = ROOT / "data/splits_stage_q2_ocr_mt"
PROMPT_KEY = "ocr_mt_v1"


def _append_real(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    src = REAL_MT if REAL_MT.is_file() else REAL_OCR
    if not src.is_file():
        return train, val, test, {"n_real": 0, "real_train": 0, "real_val": 0, "real_test": 0, "real_src": ""}
    real = load_jsonl(src)
    wrapped = src == REAL_MT
    for row in real:
        meta = dict(row.get("metadata") or {})
        meta["pool_id"] = str(meta.get("pool_id") or "real_labeled")
        meta["stage"] = "stage_q2_ocr_mt"
        meta["prompt_key"] = PROMPT_KEY
        msgs = list(row.get("messages") or [])
        if msgs and msgs[-1].get("role") == "assistant" and not wrapped:
            md = str(msgs[-1].get("content") or "")
            zh = str(meta.get("translation") or "")
            msgs[-1] = {**msgs[-1], "content": wrap_point_target(md, prompt_key=PROMPT_KEY, translation=zh)}
            row["messages"] = msgs
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
        "real_src": str(src),
    }


def main() -> None:
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
        POOLS_Q2,
        rng=rng,
        prompt_key=PROMPT_KEY,
        chrome_frac=float(recipe["chrome_frac"]),
    )
    for row in rows:
        meta = dict(row.get("metadata") or {})
        meta["stage"] = "stage_q2_ocr_mt"
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
        "name": "stage_q2_ocr_mt",
        "prompt_key": PROMPT_KEY,
        "marker": "x45d_mix_0.5_0.85",
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
    print(json.dumps({k: meta[k] for k in ("n_train", "n_val", "n_test", "synthetic_quotas", "real_train", "real_src", "train") if k in meta}, indent=2), flush=True)


if __name__ == "__main__":
    main()

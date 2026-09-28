#!/usr/bin/env python3
"""Compose the GRPO_Q1 v2 split from the value probe + a balanced SFT val.

Why the old ``max R > 0.5`` rule no longer works
------------------------------------------------
The v2 reward has *different ranges per scene*:

* positive: ceiling = ``edit`` = **1.0**  (empty/over/under only subtract)
* negative: ceiling = ``edit`` + ``empty`` = **1.0 + 1.5 = 2.5** (empty↔empty)

So a fixed ``0.5`` means "half-correct block" on a positive but "a quarter of an
empty rollout" on a negative.  Instead this script gates on a **fraction of the
scene ceiling** and then *ranks* by the failure mode we want GRPO to fix.

Selection per record (G rollouts, scored with the q1v2 reward):

1. ``good``  = reward >= ``good_frac`` × ceiling   (default 0.8)
2. hard gate = ``min_good_frac <= frac_good <= max_good_frac`` and
   ``reward_std >= min_std``:
   at least one good rollout (a trajectory to reinforce) AND at least one bad
   one (variance -> advantage).  Fully saturated / fully collapsed groups are
   dropped because their advantage is zero.
3. value score (used to pick the top N inside each scene quota):
     - positives: ``frac_over + frac_under + 0.5*frac_empty_miss``
       i.e. prefer groups where the model actually **over-dumps** or **misses
       members** — the multi_frag / short-group boundary.
     - negatives: ``1 - 2*|frac_good - 0.5|``
       prefer groups that are genuinely split between empty and hallucination.

  uv run python eval/run_grpo_value_probe.py \
    --src data/splits_stage_q1_withreal/train.jsonl \
    --out checkpoints/grpo_value_probe_v2 --reward-set q1v2 \
    --scenes regular,multi_frag,semantic_group --include-empty
  uv run python data/scripts/compose_grpo_q1_v2.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import load_jsonl  # noqa: E402
from point_ocr.grpo_data import (  # noqa: E402
    _meta,
    _tag_row,
    interleave_stage,
    select_sliced_rows,
    val_slice,
)
from point_ocr.grpo_rewards import group_reward_stats  # noqa: E402
from point_ocr.grpo_rewards_q1 import (  # noqa: E402
    DEFAULT_WEIGHTS,
    reward_breakdown,
    weighted_rewards,
)
from point_ocr.pools.compose import mix_summary, write_rows  # noqa: E402

DEFAULT_RECIPE = ROOT / "data/recipes/grpo_q1_v2.yaml"
_CEILING = {
    "positive": DEFAULT_WEIGHTS.edit,
    "negative": DEFAULT_WEIGHTS.edit + DEFAULT_WEIGHTS.empty,
}


def _load_probe(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def score_record(rec: dict[str, Any], *, good_frac: float) -> dict[str, Any]:
    """Rescore a probe record with the q1v2 reward and summarise the failure modes."""
    preds = [str(p or "") for p in (rec.get("completions") or [])]
    gt = str(rec.get("target") or "")
    neg = bool(rec.get("is_negative"))
    out = dict(rec)
    if not preds:
        out.update({"n": 0, "frac_good": 0.0, "reward_std": 0.0, "reward_max": 0.0, "value": 0.0})
        return out
    rewards = weighted_rewards(preds, gt, is_negative=neg)
    stats = group_reward_stats(rewards)
    bd = reward_breakdown(preds, gt, is_negative=neg)
    n = len(preds)
    ceiling = _CEILING["negative"] if neg else _CEILING["positive"]
    frac_good = sum(1 for r in rewards if r >= good_frac * ceiling) / n
    frac_over = sum(1 for b in bd if b["over_extraction"] < 0) / n
    frac_under = sum(1 for b in bd if b["under_extraction"] < 0) / n
    frac_empty_miss = sum(1 for b in bd if b["empty"] < 0) / n
    if neg:
        value = 1.0 - 2.0 * abs(frac_good - 0.5)
    else:
        value = frac_over + frac_under + 0.5 * frac_empty_miss
    out.update(
        {
            "rewards": rewards,
            "reward_mean": stats["mean"],
            "reward_std": stats["std"],
            "reward_min": stats["min"],
            "reward_max": stats["max"],
            "ceiling": ceiling,
            "frac_good": frac_good,
            "frac_over": frac_over,
            "frac_under": frac_under,
            "frac_empty_miss": frac_empty_miss,
            "n": n,
            "value": value,
        }
    )
    return out


def is_selectable(rec: dict[str, Any], *, min_good_frac: float, max_good_frac: float, min_std: float) -> bool:
    fg = float(rec.get("frac_good") or 0.0)
    return (
        min_good_frac <= fg <= max_good_frac
        and float(rec.get("reward_std") or 0.0) >= min_std
    )


def compose(
    *,
    recipe: dict[str, Any],
    src_dir: Path,
    probe_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    seed = int(recipe.get("seed") or 42)
    stage = str(recipe.get("name") or "grpo_q1_v2")
    mix = {str(k): float(v) for k, v in (recipe.get("mix") or {}).items()}
    val_mix = {str(k): float(v) for k, v in (recipe.get("val_mix") or mix).items()}
    good_frac = float(recipe.get("good_frac", 0.8))
    min_good_frac = float(recipe.get("min_good_frac", 0.125))
    max_good_frac = float(recipe.get("max_good_frac", 0.875))
    min_std = float(recipe.get("min_std", 0.03))
    empty_chrome_frac = recipe.get("empty_chrome_frac")
    empty_chrome_frac = float(empty_chrome_frac) if empty_chrome_frac is not None else None

    probe = [score_record(r, good_frac=good_frac) for r in _load_probe(probe_path)]
    train_src = load_jsonl(src_dir / "train.jsonl")
    by_id = {str(_meta(r).get("sample_id") or ""): r for r in train_src}

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    for rec in probe:
        scene = str(rec.get("scene") or "")
        if scene not in mix:
            continue
        counts[f"{scene}:total"] += 1
        if not is_selectable(rec, min_good_frac=min_good_frac, max_good_frac=max_good_frac, min_std=min_std):
            continue
        counts[f"{scene}:selectable"] += 1
        row = by_id.get(str(rec.get("sample_id") or ""))
        if row is not None:
            rec = dict(rec)
            rec["_row"] = row
            by_scene[scene].append(rec)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    picked_counts: dict[str, int] = {}
    for scene, frac in mix.items():
        want = int(round(frac * int(recipe["n_train"])))
        pool = list(by_scene.get(scene) or [])
        # Rank by value (boundary failures / split groups), tie-break by std.
        rng.shuffle(pool)
        pool.sort(key=lambda r: (float(r.get("value") or 0.0), float(r.get("reward_std") or 0.0)), reverse=True)
        take = min(want, len(pool))
        picked_counts[scene] = take
        train.extend(_tag_row(r["_row"], scene=scene, stage=stage) for r in pool[:take])

    n_val = int(recipe.get("n_val") or 0)
    val: list[dict[str, Any]] = []
    if n_val > 0 and (src_dir / "val.jsonl").is_file():
        val_src = load_jsonl(src_dir / "val.jsonl")
        val = select_sliced_rows(
            val_src,
            n_val,
            val_mix,
            random.Random(seed + 1),
            slice_fn=val_slice,
            stage=stage,
            empty_chrome_frac=empty_chrome_frac,
        )

    train = interleave_stage(train, rng)
    val = interleave_stage(val, random.Random(seed + 2))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": stage,
        "from": str(src_dir),
        "probe": str(probe_path),
        "prompt_key": recipe.get("prompt_key") or "a2_v3",
        "reward_set": "q1v2",
        "criteria": {
            "good_frac": good_frac,
            "min_good_frac": min_good_frac,
            "max_good_frac": max_good_frac,
            "min_std": min_std,
            "ceiling": _CEILING,
        },
        "n_train": write_rows(out_dir / "train.jsonl", train),
        "n_val": write_rows(out_dir / "val.jsonl", val) if val else 0,
        "mix": mix,
        "val_mix": val_mix,
        "probe_counts": dict(counts),
        "picked_by_scene": picked_counts,
        "train": mix_summary(train),
        "val": mix_summary(val) if val else {},
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Compose GRPO_Q1 v2 split")
    ap.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--probe", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    src_dir = Path(args.src) if args.src else ROOT / str(recipe["from"])
    probe_path = Path(args.probe) if args.probe else ROOT / str(recipe["probe"])
    out_dir = Path(args.out) if args.out else ROOT / str(recipe["out"])

    meta = compose(recipe=recipe, src_dir=src_dir, probe_path=probe_path, out_dir=out_dir)
    keys = ("n_train", "n_val", "picked_by_scene", "probe_counts", "criteria")
    print(json.dumps({k: meta[k] for k in keys if k in meta}, indent=2, ensure_ascii=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

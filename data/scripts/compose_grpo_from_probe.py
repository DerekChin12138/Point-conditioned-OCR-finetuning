#!/usr/bin/env python3
"""Build the next GRPO split from the value-probe jsonl + a balanced SFT val.

Rescores probe completions with the current GRPO reward (edit + empty-miss + leak).
Keeps signal rows that still have a usable trajectory (max R >= min_max_reward).
Val is 25% regular / multi_frag / semantic_group / empty from the SFT val split.

  uv run python data/scripts/compose_grpo_from_probe.py
  uv run python data/scripts/compose_grpo_from_probe.py --min-probe 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
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
    is_real_row,
    select_sliced_rows,
    val_slice,
)
from point_ocr.grpo_rewards import (  # noqa: E402
    group_learning_verdict,
    group_reward_stats,
    weighted_rewards,
)
from point_ocr.pools.compose import mix_summary, write_rows  # noqa: E402

DEFAULT_RECIPE = ROOT / "data/recipes/grpo_q1_signal.yaml"


def _load_probe(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def rescore_probe_record(rec: dict[str, Any], *, min_std: float) -> dict[str, Any]:
    preds = [str(p or "") for p in (rec.get("completions") or [])]
    gt = str(rec.get("target") or "")
    if not preds:
        out = dict(rec)
        out["verdict"] = "flat"
        out["rewards"] = []
        return out
    rewards = weighted_rewards(
        preds, gt, is_negative=bool(rec.get("is_negative"))
    )
    stats = group_reward_stats(rewards)
    n_unique = len(list(dict.fromkeys(preds)))
    out = dict(rec)
    out["rewards"] = rewards
    out["reward_mean"] = stats["mean"]
    out["reward_std"] = stats["std"]
    out["reward_min"] = stats["min"]
    out["reward_max"] = stats["max"]
    out["n_unique"] = n_unique
    out["verdict"] = group_learning_verdict(
        stats["mean"], stats["std"], n_unique, min_std=min_std
    )
    return out


def is_high_value(rec: dict[str, Any], *, min_max_reward: float) -> bool:
    if rec.get("is_negative") or str(rec.get("scene") or "") == "empty":
        return False
    if rec.get("verdict") != "signal":
        return False
    return float(rec.get("reward_max") or 0.0) >= float(min_max_reward)


def compose_from_probe(
    *,
    recipe: dict[str, Any],
    src_dir: Path,
    probe_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    min_std = float(recipe.get("min_std") or 0.04)
    min_max = float(recipe.get("min_max_reward") or 0.5)
    seed = int(recipe.get("seed") or 42)
    stage = str(recipe.get("name") or "grpo_q1_signal")

    probe = [rescore_probe_record(r, min_std=min_std) for r in _load_probe(probe_path)]
    keep_ids = {str(r.get("sample_id")) for r in probe if is_high_value(r, min_max_reward=min_max)}
    train_src = load_jsonl(src_dir / "train.jsonl")
    by_id = {str(_meta(r).get("sample_id") or ""): r for r in train_src}
    train: list[dict[str, Any]] = []
    missing = 0
    for rec in probe:
        sid = str(rec.get("sample_id") or "")
        if sid not in keep_ids:
            continue
        row = by_id.get(sid)
        if row is None:
            missing += 1
            continue
        scene = str(rec.get("scene") or val_slice(row) or "multi_frag")
        tagged = _tag_row(row, scene=scene, stage=stage)
        tagged["metadata"]["probe_verdict"] = rec.get("verdict")
        tagged["metadata"]["probe_reward_mean"] = rec.get("reward_mean")
        tagged["metadata"]["probe_reward_std"] = rec.get("reward_std")
        tagged["metadata"]["probe_reward_max"] = rec.get("reward_max")
        train.append(tagged)

    if bool(recipe.get("keep_all_real")):
        have = {str(_meta(r).get("sample_id") or "") for r in train}
        for rec in probe:
            sid = str(rec.get("sample_id") or "")
            if sid in have:
                continue
            row = by_id.get(sid)
            if row is None or not is_real_row(row):
                continue
            if not is_high_value(rec, min_max_reward=min_max):
                continue
            tagged = _tag_row(row, scene=str(rec.get("scene") or val_slice(row) or "semantic_group"), stage=stage)
            train.append(tagged)

    rng = random.Random(seed)
    train = interleave_stage(train, rng)

    n_val = int(recipe.get("n_val") or 0)
    val_mix = {str(k): float(v) for k, v in (recipe.get("val_mix") or {}).items()}
    val: list[dict[str, Any]] = []
    if n_val and val_mix and (src_dir / "val.jsonl").is_file():
        chrome = recipe.get("empty_chrome_frac")
        val = select_sliced_rows(
            load_jsonl(src_dir / "val.jsonl"),
            n_val,
            val_mix,
            random.Random(seed + 1),
            slice_fn=val_slice,
            stage=stage,
            empty_chrome_frac=float(chrome) if chrome is not None else None,
        )
        val = interleave_stage(val, random.Random(seed + 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    probe_verdicts = Counter(str(r.get("verdict") or "?") for r in probe)
    meta = {
        "name": stage,
        "from": str(src_dir),
        "probe": str(probe_path),
        "n_probe": len(probe),
        "probe_verdicts": dict(probe_verdicts),
        "n_high_value": len(keep_ids),
        "n_train_missing_src": missing,
        "min_std": min_std,
        "min_max_reward": min_max,
        "n_train": write_rows(out_dir / "train.jsonl", train),
        "n_val": write_rows(out_dir / "val.jsonl", val) if val else 0,
        "val_mix": val_mix,
        "train": mix_summary(train),
        "val": mix_summary(val) if val else {},
        "train_by_scene": dict(Counter(str(_meta(r).get("grpo_scene")) for r in train)),
        "val_by_scene": dict(Counter(str(_meta(r).get("grpo_scene")) for r in val)),
    }
    (out_dir / "split_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "probe_rescored.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in probe),
        encoding="utf-8",
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--probe", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--min-probe",
        type=int,
        default=0,
        help="Refuse to compose if probe jsonl has fewer scored rows.",
    )
    args = ap.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise SystemExit(f"recipe {args.recipe} is not a mapping")
    src = args.src or ROOT / str(recipe.get("from") or "data/splits_stage_q1_withreal")
    probe = args.probe or ROOT / str(recipe.get("probe") or "checkpoints/grpo_value_probe/rows.jsonl")
    out = args.out or ROOT / str(recipe.get("out") or "data/splits_grpo_q1_signal")
    n_probe = len(_load_probe(probe))
    if args.min_probe and n_probe < args.min_probe:
        raise SystemExit(f"probe has {n_probe} rows; need >= {args.min_probe}")
    meta = compose_from_probe(recipe=recipe, src_dir=src, probe_path=probe, out_dir=out)
    print(json.dumps({k: meta[k] for k in (
        "name", "n_probe", "probe_verdicts", "n_high_value", "n_train", "n_val",
        "train_by_scene", "val_by_scene",
    ) if k in meta}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

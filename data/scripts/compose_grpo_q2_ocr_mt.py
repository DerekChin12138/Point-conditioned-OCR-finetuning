#!/usr/bin/env python3
"""Compose the GRPO_Q2 (OCR + EN→ZH) split from the value probe.

Same shape as ``compose_grpo_q1_v2.py`` but Q2 has **two** quality axes, so
"good" is defined on both instead of a single reward threshold:

* positive: ``edit_sim(pred_source, gt_source) >= source_threshold``
  **and** ``chrf(pred_translation, gt_translation) >= translation_threshold``
* negative (empty GT): the prediction is effectively empty

Hard gate: ``min_good_frac <= frac_good <= max_good_frac`` and ``reward_std >= min_std``
(some rollouts succeed, some fail -> non-zero advantage).

Value ranking inside each scene quota:
* positives: ``frac_over_extraction + frac_bad_translation + 0.5 * frac_empty_miss``
* negatives: ``1 - 2 * |frac_good - 0.5|``

  uv run python eval/run_grpo_value_probe.py \
    --src data/splits_stage_q2_ocr_mt_v2/train.jsonl \
    --out checkpoints/grpo_value_probe_q2 --reward-set q2 \
    --scenes regular,multi_frag,semantic_group --include-empty
  uv run python data/scripts/compose_grpo_q2_ocr_mt.py
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
from point_ocr.grpo_rewards_q2 import reward_breakdown, weighted_rewards_q2  # noqa: E402
from point_ocr.metrics import edit_similarity  # noqa: E402
from point_ocr.pools.compose import mix_summary, write_rows  # noqa: E402
from point_ocr.q2_metrics import chrf, parse_ocr_mt_prediction  # noqa: E402

DEFAULT_RECIPE = ROOT / "data/recipes/grpo_q2_ocr_mt.yaml"


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


def score_record(
    rec: dict[str, Any],
    *,
    source_threshold: float,
    translation_threshold: float,
    bad_translation_chrf: float,
) -> dict[str, Any]:
    preds = [str(p or "") for p in (rec.get("completions") or [])]
    gt = str(rec.get("target") or "")
    neg = bool(rec.get("is_negative"))
    out = dict(rec)
    if not preds:
        out.update({"n": 0, "frac_good": 0.0, "reward_std": 0.0, "value": 0.0})
        return out
    g = parse_ocr_mt_prediction(gt)
    rewards = weighted_rewards_q2(preds, gt, is_negative=neg)
    stats = group_reward_stats(rewards)
    bd = reward_breakdown(preds, gt, is_negative=neg)
    n = len(preds)
    n_good = n_over = n_bad_tr = n_empty_miss = 0
    for pred, b in zip(preds, bd):
        p = parse_ocr_mt_prediction(pred)
        if neg:
            ok = (not pred.strip()) or p.empty_shell or p.truncated
        else:
            ok = (
                p.has_source
                and edit_similarity(p.source, g.source) >= source_threshold
                and chrf(p.translation, g.translation) >= translation_threshold
            )
            if b["over_extraction"] < 0:
                n_over += 1
            if chrf(p.translation, g.translation) < bad_translation_chrf:
                n_bad_tr += 1
            if b["empty"] < 0:
                n_empty_miss += 1
        n_good += int(ok)
    frac_good = n_good / n
    if neg:
        value = 1.0 - 2.0 * abs(frac_good - 0.5)
    else:
        value = (n_over + n_bad_tr + 0.5 * n_empty_miss) / n
    out.update(
        {
            "rewards": rewards,
            "reward_mean": stats["mean"],
            "reward_std": stats["std"],
            "reward_max": stats["max"],
            "frac_good": frac_good,
            "frac_over": n_over / n,
            "frac_bad_translation": n_bad_tr / n,
            "n": n,
            "value": value,
        }
    )
    return out


def is_selectable(rec: dict[str, Any], *, min_good_frac: float, max_good_frac: float, min_std: float) -> bool:
    fg = float(rec.get("frac_good") or 0.0)
    return min_good_frac <= fg <= max_good_frac and float(rec.get("reward_std") or 0.0) >= min_std


def _scene_of(row: dict[str, Any]) -> str | None:
    if str(_meta(row).get("pool_id") or "") == "real_labeled":
        return "real"
    scene = val_slice(row)
    return scene if scene in {"regular", "multi_frag", "semantic_group", "empty"} else None


def compose(*, recipe: dict[str, Any], src_dir: Path, probe_path: Path, out_dir: Path) -> dict[str, Any]:
    seed = int(recipe.get("seed") or 42)
    stage = str(recipe.get("name") or "grpo_q2_ocr_mt")
    mix = {str(k): float(v) for k, v in (recipe.get("mix") or {}).items()}
    val_mix = {str(k): float(v) for k, v in (recipe.get("val_mix") or mix).items()}
    src_thr = float(recipe.get("source_threshold", 0.85))
    tr_thr = float(recipe.get("translation_threshold", 0.5))
    bad_tr = float(recipe.get("bad_translation_chrf", 0.3))
    min_good_frac = float(recipe.get("min_good_frac", 0.125))
    max_good_frac = float(recipe.get("max_good_frac", 0.875))
    min_std = float(recipe.get("min_std", 0.03))

    probe = [
        score_record(r, source_threshold=src_thr, translation_threshold=tr_thr, bad_translation_chrf=bad_tr)
        for r in _load_probe(probe_path)
    ]
    train_src = load_jsonl(src_dir / "train.jsonl")
    by_id = {str(_meta(r).get("sample_id") or ""): r for r in train_src}
    id_scene = {sid: _scene_of(row) for sid, row in by_id.items()}

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    for rec in probe:
        sid = str(rec.get("sample_id") or "")
        scene = id_scene.get(sid) or str(rec.get("scene") or "")
        if scene not in mix:
            continue
        counts[f"{scene}:total"] += 1
        if not is_selectable(rec, min_good_frac=min_good_frac, max_good_frac=max_good_frac, min_std=min_std):
            continue
        counts[f"{scene}:selectable"] += 1
        row = by_id.get(sid)
        if row is not None:
            rec = dict(rec)
            rec["_row"] = row
            by_scene[scene].append(rec)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    picked: dict[str, int] = {}
    for scene, frac in mix.items():
        want = int(round(frac * int(recipe["n_train"])))
        pool = list(by_scene.get(scene) or [])
        rng.shuffle(pool)
        pool.sort(key=lambda r: (float(r.get("value") or 0.0), float(r.get("reward_std") or 0.0)), reverse=True)
        take = min(want, len(pool))
        picked[scene] = take
        train.extend(_tag_row(r["_row"], scene=scene, stage=stage) for r in pool[:take])

    n_val = int(recipe.get("n_val") or 0)
    val: list[dict[str, Any]] = []
    if n_val > 0 and (src_dir / "val.jsonl").is_file():
        val = select_sliced_rows(
            load_jsonl(src_dir / "val.jsonl"),
            n_val,
            val_mix,
            random.Random(seed + 1),
            slice_fn=val_slice,
            stage=stage,
        )

    train = interleave_stage(train, rng)
    val = interleave_stage(val, random.Random(seed + 2))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": stage,
        "from": str(src_dir),
        "probe": str(probe_path),
        "prompt_key": recipe.get("prompt_key") or "ocr_mt_v1",
        "reward_set": "q2",
        "criteria": {
            "source_threshold": src_thr,
            "translation_threshold": tr_thr,
            "bad_translation_chrf": bad_tr,
            "min_good_frac": min_good_frac,
            "max_good_frac": max_good_frac,
            "min_std": min_std,
        },
        "n_train": write_rows(out_dir / "train.jsonl", train),
        "n_val": write_rows(out_dir / "val.jsonl", val) if val else 0,
        "mix": mix,
        "val_mix": val_mix,
        "probe_counts": dict(counts),
        "picked_by_scene": picked,
        "train": mix_summary(train),
        "val": mix_summary(val) if val else {},
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Compose GRPO_Q2 OCR-MT split")
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

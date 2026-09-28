#!/usr/bin/env python3
"""Turn a GRPO candidate pool (+ value probe) into the final HQ-200 split.

Two modes:

  * **probe mode** (default): keep only rows whose G rollouts show *learning
    signal* — at least one good trajectory AND at least one bad one:
    ``0.125 <= frac_good <= 0.875`` and ``reward_std >= 0.03`` (q1v2 reward).
    Rank the survivors by the failure mode we want fixed
    (``frac_over + frac_under + 0.5*frac_empty_miss``), then take a per-reason
    quota. This is what makes the data "ultra high quality": every row is a
    decision the SFT model gets right *sometimes*.

  * **seed mode** (`--probe none`): no GPU needed; rank by the builder's
    ``grp_score`` heuristic. Signal is unverified — use it to smoke the GRPO
    pipeline, then re-run with the probe.

Post-gate diversity guards (per reason, so buckets cannot starve each other):
``--page-cap`` (default 2), template guard 35%, marker-area tertiles balanced.

  # after: build_grpo_candidates.py  →  run_grpo_value_probe.py
  uv run python data/scripts/compose_grpo_q1_hq200.py \
    --cand data/splits_grpo_cand --probe checkpoints/grpo_value_probe_hq/rows.jsonl \
    --out data/splits_grpo_q1_hq200 --n-train 200 --n-val 60
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import load_jsonl  # noqa: E402
from point_ocr.grpo_data import _meta  # noqa: E402
from point_ocr.grpo_rewards import group_reward_stats  # noqa: E402
from point_ocr.grpo_rewards_q1 import (  # noqa: E402
    DEFAULT_WEIGHTS as Q1_WEIGHTS,
    reward_breakdown as breakdown_q1,
    weighted_rewards as weighted_q1,
)
from point_ocr.grpo_rewards_q2 import (  # noqa: E402
    positive_ceiling,
    reward_breakdown as breakdown_q2,
    weights_from_env,
    weighted_rewards_q2,
)
from point_ocr.pools.compose import interleave_stage, write_rows  # noqa: E402

# final-200 reason quotas (hard buckets first)
HQ_MIX = {
    "multi_frag": 0.35,
    "real": 0.25,
    "semantic_group": 0.15,
    "long_block": 0.125,
    "repeat_risk": 0.125,
}
SCENE_OF = {
    "multi_frag": "multi_frag",
    "semantic_group": "semantic_group",
    "real": "regular",
    "long_block": "regular",
    "repeat_risk": "regular",
}
CEILING = float(Q1_WEIGHTS.edit)  # q1 positives only (ceiling 1.0)
CAND_BY_KIND = {"q1": "data/splits_grpo_cand", "q2": "data/splits_grpo_cand_q2"}
OUT_BY_KIND = {"q1": "data/splits_grpo_q1_hq200", "q2": "data/splits_grpo_q2_hq200"}
PROMPT_BY_KIND = {"q1": "a2_v3", "q2": "ocr_mt_v1"}
STAGE = "grpo_q1_hq200"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def score_probe(rec: dict[str, Any], *, good_frac: float, kind: str = "q1") -> dict[str, Any]:
    """Re-score a probe record with the training reward of the matching task."""
    preds = [str(p or "") for p in (rec.get("completions") or [])]
    gt = str(rec.get("target") or "")
    if not preds or bool(rec.get("is_negative")):
        return {**rec, "n": len(preds), "frac_good": 0.0, "reward_std": 0.0, "value": 0.0, "selectable": False}
    n = len(preds)
    if kind == "q2":
        w = weights_from_env()
        ceiling = positive_ceiling(w)
        rewards = weighted_rewards_q2(preds, gt, is_negative=False, weights=w)
        bd = breakdown_q2(preds, gt, is_negative=False, weights=w)
        # localisation/format-first value: over-extraction, missed block, broken XML
        frac_over = sum(1 for b in bd if b["over_extraction"] < 0) / n
        frac_miss = sum(1 for b in bd if b["empty"] < 0) / n
        frac_badxml = sum(1 for b in bd if b["xml"] <= 0) / n
        value = frac_over + 0.5 * frac_miss + 0.5 * frac_badxml
        extra = {"frac_over": frac_over, "frac_under": frac_miss, "frac_badxml": frac_badxml}
    else:
        ceiling = CEILING
        rewards = weighted_q1(preds, gt, is_negative=False)
        bd = breakdown_q1(preds, gt, is_negative=False)
        frac_over = sum(1 for b in bd if b["over_extraction"] < 0) / n
        frac_miss = sum(1 for b in bd if b["empty"] < 0) / n
        value = frac_over + frac_miss + 0.5 * frac_miss
        extra = {"frac_over": frac_over, "frac_under": frac_miss, "frac_badxml": 0.0}
    stats = group_reward_stats(rewards)
    frac_good = sum(1 for r in rewards if r >= good_frac * ceiling) / n
    rmax, rmin = max(rewards), min(rewards)
    return {
        **rec,
        "n": n,
        "rewards": rewards,
        "reward_mean": stats["mean"],
        "reward_std": stats["std"],
        "reward_max": rmax,
        "reward_min": rmin,
        "reward_max_frac": rmax / ceiling if ceiling else 0.0,
        "reward_min_frac": rmin / ceiling if ceiling else 0.0,
        "ceiling": ceiling,
        "frac_good": frac_good,
        "value": value,
        **extra,
        "selectable": True,
    }


def is_selectable(
    rec: dict[str, Any],
    *,
    min_good_frac: float,
    max_good_frac: float,
    min_std: float,
    min_reward_max_frac: float = 0.0,
) -> bool:
    """Signal gate.

    A group is worth GRPO only when it has a **near-perfect trajectory to
    reinforce** (``reward_max_frac >= min_reward_max_frac``) **and** real spread
    (``reward_std >= min_std``), while not being fully saturated
    (``frac_good <= max_good_frac``). All-good / all-bad groups have zero
    advantage and are dropped.
    """
    if not rec.get("selectable"):
        return False
    fg = float(rec.get("frac_good") or 0.0)
    rmax = float(rec.get("reward_max_frac") or 0.0)
    return (
        min_good_frac <= fg <= max_good_frac
        and float(rec.get("reward_std") or 0.0) >= min_std
        and rmax >= min_reward_max_frac
    )


def pick_two_per_page(cands: list[dict[str, Any]], n: int, *, page_cap: int = 2) -> list[dict[str, Any]]:
    """Rank by signal value and accept with per-bucket diversity guards.

    Guards: <=`page_cap` rows per page, a fair-share template cap, and a balanced
    share of the three marker-area tertiles. Tertile balance is enforced with an
    explicit quota per pass (ceil(n/3), then +1, then unbounded) rather than a
    round-robin: a page holds only `page_cap` slots while its candidates span all
    three tertiles, so a fixed rotation would systematically lock one out.
    """
    if n <= 0:
        return []
    rng = random.Random(1234)
    pool = list(cands)
    rng.shuffle(pool)  # tie-break: equal-value candidates in random order
    # "high quality" = a near-perfect rollout to reinforce AND real spread
    pool.sort(
        key=lambda c: (
            float(c.get("reward_max_frac") or 0.0),
            float(c.get("reward_std") or 0.0),
            float(c.get("value") or 0.0),
        ),
        reverse=True,
    )

    page_used: Counter[str] = Counter()
    tmpl_used: Counter[str] = Counter()
    tertile_used: Counter[int] = Counter()
    n_templates = len({str(_meta(c["_row"]).get("template_stem") or "?") for c in pool})
    tpl_cap = max(8, -(-int(1.2 * n) // max(1, n_templates)))
    t_cap = max(1, -(-n // 3))
    out: list[dict[str, Any]] = []
    used: set[int] = set()
    for budget in (t_cap, t_cap + 1, 10 ** 9):
        for cand in pool:
            if len(out) >= n:
                return out
            if id(cand) in used:
                continue
            row = cand["_row"]
            meta = _meta(row)
            t = int(meta.get("grp_tertile") or 0)
            if tertile_used[t] >= budget:
                continue
            page = str(meta.get("page_id") or "")
            tmpl = str(meta.get("template_stem") or "unknown")
            if page and page_used[page] >= page_cap:
                continue
            if tmpl_used[tmpl] >= tpl_cap:
                continue
            used.add(id(cand))
            page_used[page] += 1
            tmpl_used[tmpl] += 1
            tertile_used[t] += 1
            out.append(cand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("q1", "q2"), default="q1",
                    help="q1 = block-only (a2_v3, reward q1v2); q2 = <source>+<translation> (ocr_mt_v1, reward q2)")
    ap.add_argument("--cand", type=Path, default=None)
    ap.add_argument("--probe", type=str, default="", help="rows.jsonl path, or 'none' for seed mode")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--good-frac", type=float, default=0.80)
    ap.add_argument("--min-good-frac", type=float, default=0.125)
    ap.add_argument("--max-good-frac", type=float, default=0.875)
    ap.add_argument("--min-std", type=float, default=0.10,
                    help="组内 reward 标准差下限（越大越挑「有高有低」的组）")
    ap.add_argument("--min-reward-max-frac", type=float, default=0.95,
                    help="组内最高 reward / ceiling 的下限（要求有一条近乎满分的轨迹）")
    args = ap.parse_args()
    args.kind = str(args.kind)
    args.cand = args.cand or ROOT / CAND_BY_KIND[args.kind]
    args.out = args.out or ROOT / OUT_BY_KIND[args.kind]

    src_train = load_jsonl(args.cand / "train.jsonl")
    by_id = {str(_meta(r).get("sample_id") or ""): r for r in src_train}
    src_val = load_jsonl(args.cand / "val.jsonl")
    if not src_train:
        raise SystemExit(f"no candidates at {args.cand}/train.jsonl — run build_grpo_candidates.py first")

    probe_mode = args.probe.strip().lower() not in {"", "none", "no", "seed"}
    counts: Counter[str] = Counter()
    scored: list[dict[str, Any]] = []
    if probe_mode:
        rows = _load_jsonl(Path(args.probe))
        counts["probe_rows"] = len(rows)
        for rec in rows:
            sc = score_probe(rec, good_frac=args.good_frac, kind=args.kind)
            counts[f"{rec.get('scene') or '?'}:probe"] += 1
            if not is_selectable(sc, min_good_frac=args.min_good_frac, max_good_frac=args.max_good_frac,
                                 min_std=args.min_std, min_reward_max_frac=args.min_reward_max_frac):
                continue
            counts[f"{rec.get('scene') or '?'}:selectable"] += 1
            row = by_id.get(str(rec.get("sample_id") or ""))
            if row is not None:
                sc["_row"] = row
                scored.append(sc)
    else:
        for row in src_train:
            meta = _meta(row)
            if bool(meta.get("is_negative")):
                continue
            scored.append({"value": float(meta.get("grp_score") or 0.0), "reward_std": 0.0, "_row": row})

    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in scored:
        reason = str(_meta(c["_row"]).get("grp_reason") or "")
        if reason in HQ_MIX:
            by_reason[reason].append(c)

    train: list[dict[str, Any]] = []
    picked: dict[str, int] = {}
    for reason, frac in HQ_MIX.items():
        want = int(round(frac * args.n_train))
        chosen = pick_two_per_page(list(by_reason.get(reason) or []), want)
        picked[reason] = len(chosen)
        for c in chosen:
            row = dict(c["_row"])
            meta = dict(_meta(row))
            meta["grpo_scene"] = SCENE_OF[reason]
            meta["stage"] = STAGE
            meta["is_negative"] = False
            if probe_mode:
                meta["grpo_signal"] = {
                    "frac_good": round(float(c.get("frac_good") or 0.0), 4),
                    "reward_std": round(float(c.get("reward_std") or 0.0), 4),
                    "value": round(float(c.get("value") or 0.0), 4),
                }
            row["metadata"] = meta
            train.append(row)

    # fill any shortfall from the other reasons (thin supply: repeat_risk)
    if len(train) < args.n_train:
        have = {str(_meta(r).get("sample_id") or "") for r in train}
        extra = [c for r in HQ_MIX for c in by_reason.get(r, []) if str(_meta(c["_row"]).get("sample_id")) not in have]
        for c in pick_two_per_page(extra, args.n_train - len(train)):
            reason = str(_meta(c["_row"]).get("grp_reason") or "long_block")
            row = dict(c["_row"])
            meta = dict(_meta(row))
            meta["grpo_scene"] = SCENE_OF.get(reason, "regular")
            meta["stage"] = STAGE
            meta["is_negative"] = False
            row["metadata"] = meta
            train.append(row)

    rng = random.Random(args.seed)
    train = interleave_stage(train, rng)

    val: list[dict[str, Any]] = []
    if args.n_val > 0 and src_val:
        val_pool = list(src_val)
        rng.shuffle(val_pool)
        per = max(1, args.n_val // max(1, len(HQ_MIX)))
        seen: set[str] = set()

        def _add_val(row: dict[str, Any], reason: str) -> None:
            r = dict(row)
            meta = dict(_meta(r))
            meta["grpo_scene"] = SCENE_OF.get(reason, "regular")
            meta["stage"] = STAGE
            meta["is_negative"] = False
            r["metadata"] = meta
            val.append(r)
            seen.add(str(meta.get("sample_id") or ""))

        for reason in HQ_MIX:
            got = 0
            for row in val_pool:
                if got >= per or len(val) >= args.n_val:
                    break
                if str(_meta(row).get("grp_reason") or "") != reason:
                    continue
                if str(_meta(row).get("sample_id") or "") in seen:
                    continue
                _add_val(row, reason)
                got += 1
        for row in val_pool:  # top up if some buckets have no held-out rows
            if len(val) >= args.n_val:
                break
            sid = str(_meta(row).get("sample_id") or "")
            if sid in seen:
                continue
            _add_val(row, str(_meta(row).get("grp_reason") or "long_block"))
        val = interleave_stage(val, random.Random(args.seed + 1))

    args.out.mkdir(parents=True, exist_ok=True)
    n_train = write_rows(args.out / "train.jsonl", train)
    n_val = write_rows(args.out / "val.jsonl", val) if val else 0
    meta = {
        "name": STAGE if args.kind == "q1" else STAGE.replace("q1", "q2"),
        "kind": args.kind,
        "mode": "probe" if probe_mode else "seed",
        "from": str(args.cand),
        "probe": args.probe if probe_mode else None,
        "reward_set": "q1v2" if args.kind == "q1" else "q2",
        "prompt_key": PROMPT_BY_KIND[args.kind],
        "reward_weights_env": os.environ.get("GRPO_REWARD_WEIGHTS") if args.kind == "q2" else None,
        "criteria": {
            "good_frac": args.good_frac,
            "min_good_frac": args.min_good_frac,
            "max_good_frac": args.max_good_frac,
            "min_std": args.min_std,
            "min_reward_max_frac": args.min_reward_max_frac,
            "ceiling": CEILING if args.kind == "q1" else f"source+translation+xml",
            "rank": "reward_max_frac, reward_std, failure-mode value",
        },
        "n_train": n_train,
        "n_val": n_val,
        "picked_by_reason": picked,
        "hq_mix": HQ_MIX,
        "probe_counts": dict(counts),
    }
    (args.out / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("mode", "n_train", "n_val", "picked_by_reason", "probe_counts")}, indent=2, ensure_ascii=False))
    print(f"[compose] wrote {args.out}/train.jsonl ({n_train}) val.jsonl ({n_val})")


if __name__ == "__main__":
    main()

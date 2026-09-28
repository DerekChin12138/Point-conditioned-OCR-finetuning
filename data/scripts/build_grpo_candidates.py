#!/usr/bin/env python3
"""Build a **high-value GRPO_Q1 candidate pool** from the SFT *train* split.

Why a separate pool: the old `compose_grpo_q1_v2` took a 503-candidate probe and
only 110 rows survived the signal gate (25%). This builder spends the budget
where the SFT actually fails:

  multi_frag     cross-fragment paragraphs      (SFT 0.69 -> 2B 0.86)
  real           real screenshots               (SFT 0.50-0.92, weakest bucket)
  semantic_group short bullet / group boundaries
  long_block     top-length core_inner blocks    (repetition / truncation risk)
  repeat_risk    targets with repeated lines/♫   (degenerate decode risk)

Diversity rules (all counters are **per bucket**, so an earlier bucket cannot
starve a later one):

  * per-page cap: adaptive `ceil(want / distinct_pages)` clamped to [2, 8].
    A rendered page hosts ~20-30 different block tasks, so a fixed 2/page cap
    is far too strict (measured: only 456/1500 selectable).
  * per-template guard: 35% of target — multi_frag only has 2 templates, so a
    tighter cap would starve it.
  * marker-area tertiles balanced inside each bucket (0.08-0.12 / 0.12-0.16 /
    0.16-0.2 %) so GRPO does not become a single-size specialist.
  * dHash near-duplicate filter on the marked-region crop (hamming <= 4).

Outputs `data/splits_grpo_cand/{train,val}.jsonl` (+ split_meta.json). Probe the
train file, then gate it with `data/scripts/compose_grpo_q1_hq200.py`.

  uv run python data/scripts/build_grpo_candidates.py
  uv run python data/scripts/build_grpo_candidates.py --target 1500 --no-dedup
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import load_jsonl  # noqa: E402
from point_ocr.grpo_data import _meta, assistant_text  # noqa: E402
from point_ocr.pools.compose import interleave_stage, write_rows  # noqa: E402
from point_ocr.q2_metrics import parse_ocr_mt_prediction  # noqa: E402

SRC_BY_KIND = {
    "q1": ROOT / "data/splits_stage_q1_withreal",      # target = Markdown block, prompt a2_v3
    "q2": ROOT / "data/splits_stage_q2_ocr_mt_v2",     # target = <source>+<translation>, ocr_mt_v1
}
OUT_BY_KIND = {
    "q1": ROOT / "data/splits_grpo_cand",
    "q2": ROOT / "data/splits_grpo_cand_q2",
}
SRC = SRC_BY_KIND["q1"]
OUT = OUT_BY_KIND["q1"]


def gt_text(row: dict[str, Any], kind: str = "q1") -> str:
    """Task target: whole assistant reply (q1) or just the `<source>` block (q2)."""
    gt = assistant_text(row)
    if kind == "q2":
        return parse_ocr_mt_prediction(gt).source
    return gt

DEFAULT_MIX = {
    "multi_frag": 0.30,
    "real": 0.26,
    "semantic_group": 0.16,
    "long_block": 0.16,
    "repeat_risk": 0.12,
}
REASON_BASE = {
    "real": 1.00,
    "multi_frag": 0.90,
    "semantic_group": 0.70,
    "repeat_risk": 0.60,
    "long_block": 0.50,
}
PAGE_CAP_MIN = 2
PAGE_CAP_MAX = 8
TEMPLATE_FRAC = 0.35
DHASH_HAMMING = 4


# --------------------------------------------------------------------------- #
# classification / scoring
# --------------------------------------------------------------------------- #
def _repeat_risk(gt: str) -> bool:
    text = gt.strip()
    if "♫" in text:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) != len(set(lines)):
        return True
    if text.count("|") >= 6:  # table-ish block
        return True
    grams: Counter[str] = Counter()
    for i in range(0, max(0, len(text) - 24)):
        gram = text[i : i + 24]
        # low-entropy stretches ("aaaa", "----", "....") repeat trivially; only
        # flag grams that carry real content.
        if len(set(gram) - set(" \t")) >= 3:
            grams[gram] += 1
    return any(c >= 3 for c in grams.values())


def reason_of(row: dict[str, Any], *, long_threshold: int, kind: str = "q1") -> str | None:
    meta = _meta(row)
    pool = str(meta.get("pool_id") or "")
    gt = gt_text(row, kind).strip()
    n_frag = int(meta.get("n_fragments") or 1)
    n_box = len(meta.get("bboxes") or [])
    if not gt:
        return None
    if pool == "real_labeled":
        return "real"
    if pool == "multi_frag" or n_frag > 1 or n_box > 1:
        return "multi_frag"
    if pool == "semantic_group":
        return "semantic_group"
    if pool == "core_inner":
        if _repeat_risk(gt):
            return "repeat_risk"
        if len(gt) >= long_threshold:
            return "long_block"
    return None


def hard_score(row: dict[str, Any], reason: str, *, tertile: int, kind: str = "q1") -> float:
    meta = _meta(row)
    gt = gt_text(row, kind).strip()
    n_frag = int(meta.get("n_fragments") or 1)
    s = REASON_BASE.get(reason, 0.4)
    s += 0.10 * max(0, n_frag - 1)
    s += 0.15 * min(1.0, len(gt) / 800.0)
    if str(meta.get("chrome_family") or "none").lower() not in {"", "none"}:
        s += 0.15  # chrome pages are the fragile ones
    if tertile == 0:
        s += 0.10  # smallest markers are hardest
    return round(s, 4)


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def crop_dhash(img_path: str, bbox: list[float] | None, *, hash_size: int = 8) -> int | None:
    """dHash of the marked block region (9x8 grayscale horizontal gradients)."""
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(img_path) as im:
            im = im.convert("L")
            w, h = im.size
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = (float(v) for v in bbox)
                pad_x, pad_y = 0.12 * (x2 - x1), 0.12 * (y2 - y1)
                box = (
                    max(0, int(x1 - pad_x)),
                    max(0, int(y1 - pad_y)),
                    min(w, int(x2 + pad_x)),
                    min(h, int(y2 + pad_y)),
                )
                if box[2] > box[0] and box[3] > box[1]:
                    im = im.crop(box)
            im = im.resize((hash_size + 1, hash_size))
            px = list(im.getdata())
    except Exception:
        return None
    bits = 0
    for r in range(hash_size):
        base = r * (hash_size + 1)
        for c in range(hash_size):
            bits = (bits << 1) | (1 if px[base + c] > px[base + c + 1] else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def marker_tertile(frac: float, cuts: tuple[float, float]) -> int:
    lo, hi = cuts
    if frac <= lo:
        return 0
    if frac <= hi:
        return 1
    return 2


def _tertile_cuts(rows: list[dict[str, Any]]) -> tuple[float, float]:
    vals = sorted(float(_meta(r).get("marker_area_frac") or 0.0) for r in rows)
    if not vals:
        return 0.0, 0.0
    return vals[len(vals) // 3], vals[2 * len(vals) // 3]


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def build_pool(
    rows: list[dict[str, Any]],
    *,
    target: int,
    mix: dict[str, float],
    seed: int,
    dedup: bool,
    kind: str = "q1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core = [len(gt_text(r, kind).strip()) for r in rows if str(_meta(r).get("pool_id")) == "core_inner"]
    long_threshold = 0
    if core:
        core.sort()
        long_threshold = core[int(len(core) * 0.80)]
    cuts = _tertile_cuts(rows)

    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pool_cands: list[dict[str, Any]] = []
    for r in rows:
        reason = reason_of(r, long_threshold=long_threshold, kind=kind)
        if reason is None:
            continue
        t = marker_tertile(float(_meta(r).get("marker_area_frac") or 0.0), cuts)
        cand = {"row": r, "reason": reason, "tertile": t,
                "score": hard_score(r, reason, tertile=t, kind=kind)}
        by_reason[reason].append(cand)
        pool_cands.append(cand)

    rng = random.Random(seed)
    page_used: dict[str, Counter[str]] = defaultdict(Counter)
    template_used: dict[str, Counter[str]] = defaultdict(Counter)
    tertile_used: dict[str, Counter[int]] = defaultdict(Counter)
    hashes: dict[str, list[int]] = defaultdict(list)
    accepted: list[dict[str, Any]] = []
    accepted_ids: set[int] = set()
    picked_by_reason: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    stats: dict[str, Any] = {"long_threshold": long_threshold, "tertile_cuts": list(cuts)}
    tpl_cap = max(4, int(TEMPLATE_FRAC * target))

    def reason_page_cap(reason: str, want: int) -> int:
        pool = by_reason.get(reason) or []
        n_pages = len({str(_meta(c["row"]).get("page_id") or "") for c in pool})
        stats[f"{reason}_pages"] = n_pages
        stats[f"{reason}_available"] = len(pool)
        cap = PAGE_CAP_MIN if n_pages <= 0 else max(PAGE_CAP_MIN, min(PAGE_CAP_MAX, -(-want // n_pages)))
        stats[f"{reason}_page_cap"] = cap
        return cap

    def try_accept(cand: dict[str, Any], *, per_tertile: int, tag: str, page_cap: int) -> bool:
        row = cand["row"]
        reason = cand["reason"]
        if id(row) in accepted_ids:
            return False
        t = cand["tertile"]
        meta = _meta(row)
        page = str(meta.get("page_id") or "")
        tmpl = str(meta.get("template_stem") or "unknown")
        if per_tertile > 0 and tertile_used[reason][t] >= per_tertile:
            rejected[f"{tag}:tertile"] += 1
            return False
        if page and page_used[reason][page] >= page_cap:
            rejected[f"{tag}:page"] += 1
            return False
        if template_used[reason][tmpl] >= tpl_cap:
            rejected[f"{tag}:template"] += 1
            return False
        dh = None
        if dedup:
            dh = crop_dhash(str((row.get("images") or [""])[0]), meta.get("bbox"))
            if dh is not None and any(_hamming(dh, h) <= DHASH_HAMMING for h in hashes[reason]):
                rejected[f"{tag}:dhash"] += 1
                return False
        page_used[reason][page] += 1
        template_used[reason][tmpl] += 1
        tertile_used[reason][t] += 1
        accepted_ids.add(id(row))
        if dh is not None:
            hashes[reason].append(dh)
        accepted.append(cand)
        return True

    # pass 1 — per-reason quota, marker tertiles balanced inside the bucket
    for reason, frac in mix.items():
        want = int(round(frac * target))
        pool = by_reason.get(reason) or []
        rng.shuffle(pool)
        pool.sort(key=lambda c: c["score"], reverse=True)
        cap = reason_page_cap(reason, want)
        per_tertile = max(1, want // 3)
        shortlist = pool[: max(want * 3, 20)]
        picked = 0
        for cand in shortlist:
            if picked >= want:
                break
            if try_accept(cand, per_tertile=per_tertile, tag=reason, page_cap=cap):
                picked += 1
        picked_by_reason[reason] = picked
        stats[f"{reason}_want"] = want
        stats[f"{reason}_shortlist"] = len(shortlist)

    # pass 2 — fill a shortfall: per-reason capped overflow, then anything left
    if len(accepted) < target:
        for reason, frac in mix.items():
            if len(accepted) >= target:
                break
            quota = int(round(frac * target))
            ceiling = int(quota * 1.25) + 2
            cap = max(reason_page_cap(reason, quota), PAGE_CAP_MAX)
            got = picked_by_reason[reason]
            left = [c for c in (by_reason.get(reason) or []) if id(c["row"]) not in accepted_ids]
            left.sort(key=lambda c: c["score"], reverse=True)
            for cand in left:
                if got >= ceiling or len(accepted) >= target:
                    break
                if try_accept(cand, per_tertile=max(1, quota // 2), tag=f"fill:{reason}", page_cap=cap):
                    got += 1
            picked_by_reason[reason] = got
        if len(accepted) < target:
            rest = [c for c in pool_cands if id(c["row"]) not in accepted_ids]
            rest.sort(key=lambda c: c["score"], reverse=True)
            for cand in rest:
                if len(accepted) >= target:
                    break
                if try_accept(cand, per_tertile=0, tag=f"fill:{cand['reason']}", page_cap=PAGE_CAP_MAX * 2):
                    picked_by_reason[cand["reason"]] += 1

    out: list[dict[str, Any]] = []
    for cand in accepted:
        row = dict(cand["row"])
        meta = dict(_meta(row))
        meta["grp_reason"] = cand["reason"]
        meta["grp_score"] = cand["score"]
        meta["grp_tertile"] = cand["tertile"]
        meta["grp_cand"] = True
        meta["grpo_scene"] = {
            "multi_frag": "multi_frag",
            "semantic_group": "semantic_group",
            "real": "regular",
            "long_block": "regular",
            "repeat_risk": "regular",
        }[cand["reason"]]
        meta["is_negative"] = False
        row["metadata"] = meta
        out.append(row)
    out = interleave_stage(out, random.Random(seed + 7))
    stats["picked_by_reason"] = dict(picked_by_reason)
    stats["rejected_by_cause"] = dict(rejected)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("q1", "q2"), default="q1",
                    help="q1 = block-only target (a2_v3); q2 = <source>+<translation> (ocr_mt_v1)")
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--val-size", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dedup", action="store_true", help="skip the dHash near-duplicate filter")
    args = ap.parse_args()
    args.kind = str(args.kind)
    args.src = args.src or SRC_BY_KIND[args.kind]
    args.out = args.out or OUT_BY_KIND[args.kind]

    train_src = [r for r in load_jsonl(args.src / "train.jsonl") if gt_text(r, args.kind).strip()]
    print(f"[cand] kind={args.kind} src={args.src} train positives={len(train_src)} (target={args.target})", flush=True)
    train, stats = build_pool(train_src, target=args.target, mix=DEFAULT_MIX, seed=args.seed,
                              dedup=not args.no_dedup, kind=args.kind)

    val_rows: list[dict[str, Any]] = []
    if (args.src / "val.jsonl").is_file():
        # The GRPO val split is a *readout* (never trained on). Sample-level
        # disjointness is guaranteed by coming from val.jsonl; we deliberately do
        # NOT exclude pages that also appear in train, otherwise `real` (only 64
        # screenshots) would leave the readout entirely.
        val_src = [r for r in load_jsonl(args.src / "val.jsonl") if gt_text(r, args.kind).strip()]
        val_rows, _ = build_pool(val_src, target=args.val_size, mix=DEFAULT_MIX, seed=args.seed + 1,
                                 dedup=False, kind=args.kind)

    args.out.mkdir(parents=True, exist_ok=True)
    n_train = write_rows(args.out / "train.jsonl", train)
    n_val = write_rows(args.out / "val.jsonl", val_rows) if val_rows else 0
    meta = {
        "name": f"grpo_cand_{args.kind}",
        "kind": args.kind,
        "from": str(args.src),
        "mix": DEFAULT_MIX,
        "page_cap": [PAGE_CAP_MIN, PAGE_CAP_MAX],
        "template_frac": TEMPLATE_FRAC,
        "dhash_hamming": None if args.no_dedup else DHASH_HAMMING,
        "n_train": n_train,
        "n_val": n_val,
        **stats,
    }
    (args.out / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    keep = ("_available", "_want", "_shortlist", "_pages", "_page_cap", "_by_cause", "_by_reason")
    print(json.dumps({k: v for k, v in meta.items() if k.endswith(keep) or k in {"n_train", "n_val", "long_threshold", "tertile_cuts"}}, indent=2, ensure_ascii=False))
    print(f"[cand] wrote {args.out}/train.jsonl ({n_train}) val.jsonl ({n_val})")


if __name__ == "__main__":
    main()

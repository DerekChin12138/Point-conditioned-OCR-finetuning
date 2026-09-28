#!/usr/bin/env python3
"""Remove misleading negative rows from Q2 data, and (optionally) rebalance.

Two modes:

``--mode splits`` (default)
    Filter the composed ``data/splits_stage_q2_ocr_mt/{train,val,test}.jsonl``
    and write a v2 split dir.  No re-render.  ``--blank-factor`` upsamples the
    surviving *pure-blank* negatives to restore the negative ratio.

``--mode pools``
    Filter ``data/pools_q2/<pool>/point_sharegpt.jsonl`` into a cleaned pool
    root for a later ``compose_q2_ocr_mt.py --pools-root`` run.

  uv run python data/scripts/filter_q2_negatives.py --dry-run
  uv run python data/scripts/filter_q2_negatives.py --mode splits
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.q2_filter import (  # noqa: E402
    PageGeometry,
    Q2FilterConfig,
    classify_negative,
    is_negative_row,
)

POOLS = ("core_inner", "empty_clear", "empty_special", "multi_frag", "semantic_group")
SPLITS = ("train", "val", "test")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def chrome_bucket(meta: dict[str, Any]) -> str:
    fam = str(meta.get("chrome_family") or "none").strip().lower()
    return "none" if fam in {"", "none"} else "chrome"


def _row_point(meta: dict[str, Any]) -> tuple[float, float] | None:
    p = meta.get("point")
    if not p:
        return None
    return float(p[0]), float(p[1])


def _on_chrome_or_ink(meta: dict[str, Any], geometry: PageGeometry) -> bool:
    pool = str(meta.get("pool_id") or "")
    page = str(meta.get("page_id") or "")
    pt = _row_point(meta)
    if not pool or not page or pt is None:
        return False
    scale = float(meta.get("device_scale_factor") or 1.0)
    x, y = pt
    from point_ocr.q2_filter import _inside  # local helper

    if any(_inside(x, y, r) for r in geometry.ink_rects(pool, page, scale)):
        return True
    return any(_inside(x, y, r) for _, r in geometry.chrome_bands(pool, page, scale))


def filter_rows(
    rows: list[dict[str, Any]],
    geometry: PageGeometry,
    cfg: Q2FilterConfig,
    *,
    blank_factor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    relabeled: Counter[str] = Counter()
    blank: list[dict[str, Any]] = []
    for row in rows:
        meta = dict(row.get("metadata") or {})
        if not is_negative_row(row):
            kept.append(row)
            continue
        verdict = classify_negative(meta, geometry, config=cfg)
        if verdict.drop:
            reasons[verdict.reason] += 1
            continue
        if verdict.new_region and verdict.new_region != meta.get("region"):
            relabeled[f"{meta.get('region')}->{verdict.new_region}"] += 1
            meta["region"] = verdict.new_region
            meta["region_relabeled_from_clear"] = True
            row = dict(row)
            row["metadata"] = meta
        kept.append(row)
        if str(meta.get("region")) == "neg_clear" and not _on_chrome_or_ink(meta, geometry):
            blank.append(row)
    n_before = len(rows)
    if blank_factor and blank_factor > 1.0 and blank:
        extra_n = int(round((blank_factor - 1.0) * len(blank)))
        for i in range(extra_n):
            src = blank[i % len(blank)]
            dup = json.loads(json.dumps(src))
            dup.setdefault("metadata", {})["upsampled"] = True
            kept.append(dup)
    return kept, {
        "n_before": n_before,
        "n_kept": len(kept),
        "n_blank": len(blank),
        "blank_factor": blank_factor,
        "drop_reasons": dict(reasons),
        "relabeled": dict(relabeled),
        "by_pool_after": dict(sorted(Counter(str((r.get("metadata") or {}).get("pool_id")) for r in kept).items())),
        "neg_after": sum(1 for r in kept if is_negative_row(r)),
    }


def run_splits(args: argparse.Namespace, geometry: PageGeometry, cfg: Q2FilterConfig) -> None:
    report: dict[str, Any] = {"mode": "splits", "config": cfg.__dict__, "splits": {}}
    for name in SPLITS:
        src = args.splits_dir / f"{name}.jsonl"
        if not src.is_file():
            raise SystemExit(f"missing split: {src}")
        rows = load_jsonl(src)
        bf = args.blank_factor if name == "train" else 1.0
        kept, stats = filter_rows(rows, geometry, cfg, blank_factor=bf)
        report["splits"][name] = stats
        print(f"{name}: {stats['n_before']} -> {stats['n_kept']} (drop {dict(stats['drop_reasons'])}) "
              f"blank={stats['n_blank']} factor={bf} neg={stats['neg_after']} relabeled={dict(stats['relabeled'])}", flush=True)
        if not args.dry_run:
            write_jsonl(args.out_dir / f"{name}.jsonl", kept)
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "filter_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        meta_src = args.splits_dir / "split_meta.json"
        if meta_src.is_file():
            meta = json.loads(meta_src.read_text(encoding="utf-8"))
            meta["name"] = "stage_q2_ocr_mt_v2"
            meta["filtered_from"] = str(args.splits_dir)
            meta["filter"] = cfg.__dict__
            meta["blank_factor"] = args.blank_factor
            meta["filter_report"] = {k: {"n_before": v["n_before"], "n_kept": v["n_kept"], "neg_after": v["neg_after"]} for k, v in report["splits"].items()}
            (args.out_dir / "split_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"\nwrote {args.out_dir}")


def run_pools(args: argparse.Namespace, geometry: PageGeometry, cfg: Q2FilterConfig) -> None:
    report: dict[str, Any] = {"mode": "pools", "config": cfg.__dict__, "pools": {}}
    for pool in args.pools:
        src = args.pools_root / pool / "point_sharegpt.jsonl"
        if not src.is_file():
            print(f"[skip] {src} missing", file=sys.stderr)
            continue
        rows = load_jsonl(src)
        kept, stats = filter_rows(rows, geometry, cfg, blank_factor=args.blank_factor)
        report["pools"][pool] = stats
        print(f"{pool:16s} {stats['n_before']} -> {stats['n_kept']} "
              f"(drop {dict(stats['drop_reasons'])}) relabeled={dict(stats['relabeled'])}", flush=True)
        if not args.dry_run:
            write_jsonl(args.out_root / pool / "point_sharegpt.jsonl", kept)
            meta_src = args.pools_root / pool / "pool_meta.json"
            if meta_src.is_file():
                shutil.copy2(meta_src, args.out_root / pool / "pool_meta.json")
    if not args.dry_run:
        args.out_root.mkdir(parents=True, exist_ok=True)
        (args.out_root / "filter_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.out_root}/filter_report.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter misleading Q2 negatives")
    ap.add_argument("--mode", choices=["splits", "pools"], default="splits")
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits_stage_q2_ocr_mt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/splits_stage_q2_ocr_mt_v2")
    ap.add_argument("--pools-root", type=Path, default=ROOT / "data/pools_q2")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/pools_q2_clean")
    ap.add_argument("--pools", nargs="*", default=list(POOLS))
    ap.add_argument("--max-special-chars", type=int, default=120)
    ap.add_argument("--max-special-words", type=int, default=12)
    ap.add_argument("--blank-factor", type=float, default=1.0,
                    help=">1 upsamples surviving pure-blank negatives to restore the ratio")
    ap.add_argument("--keep-chrome-clear", action="store_true")
    ap.add_argument("--drop-chrome-clear", action="store_true")
    ap.add_argument("--keep-chrome-mislabel", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = Q2FilterConfig(
        max_special_chars=args.max_special_chars,
        max_special_words=args.max_special_words,
        drop_chrome_clear=args.drop_chrome_clear,
        relabel_clear_on_chrome=not args.keep_chrome_clear,
        drop_chrome_mislabel=not args.keep_chrome_mislabel,
    )
    geometry = PageGeometry(args.pools_root)
    if args.mode == "splits":
        run_splits(args, geometry, cfg)
    else:
        run_pools(args, geometry, cfg)
    if args.dry_run:
        print("[dry-run] nothing written")


if __name__ == "__main__":
    main()

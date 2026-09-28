#!/usr/bin/env python3
"""Split ShareGPT JSONL into train / val / test (by sample, shuffled).

Default: 90% / 5% / 5%. Writes JSONL + a small manifest for notebooks.

For A2, prefer::

  --stratify-key a2_slice --interleave-template

so each split keeps slice mix, and train order is round-robin by template
(not clustered by generation order). Trainer still shuffles each epoch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.a2_mix import (  # noqa: E402
    interleave_a2_samples,
    meta_template_stem,
    summarize_mix,
)
from point_ocr.dataset_format import load_jsonl  # noqa: E402


def _stratified_split(
    rows: list[dict],
    *,
    key: str,
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        meta = row.get("metadata") or {}
        buckets[str(meta.get(key) or "_none")].append(row)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for _k, bucket in buckets.items():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        # remainder → test
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])
    return train, val, test


def _plain_split(
    rows: list[dict],
    *,
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "synth" / "point_sharegpt.jsonl",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.90)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument(
        "--stratify-key",
        type=str,
        default="",
        help="Metadata key for stratified split (e.g. a2_slice). Empty = plain shuffle.",
    )
    ap.add_argument(
        "--interleave-template",
        action="store_true",
        help="Round-robin train/val/test by (a2_slice, template_stem) after split.",
    )
    args = ap.parse_args()

    s = args.train_frac + args.val_frac + args.test_frac
    if abs(s - 1.0) > 1e-6:
        raise SystemExit(f"fractions must sum to 1, got {s}")

    if not args.input.is_file():
        raise SystemExit(
            f"missing dataset JSONL: {args.input}\n"
            "data/processed/ is gitignored — clone 不会带上训练数据。请先造数，例如：\n"
            "  uv run playwright install chromium\n"
            "  uv run playwright install-deps chromium   # Linux：缺 libgbm 等时必做，否则渲染失败无本文件\n"
            "  uv run python data/scripts/build_synth_batch.py --out data/processed/synth --noise --seed 0\n"
            "  uv run python data/scripts/expand_synth_to_n.py --target 5000\n"
            "然后再跑本脚本；或从其他机器拷贝 data/processed/synth/ 与 data/splits/。"
        )

    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit(f"empty input: {args.input}")

    rng = random.Random(args.seed)
    if args.stratify_key:
        train, val, test = _stratified_split(
            rows,
            key=args.stratify_key,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            rng=rng,
        )
    else:
        train, val, test = _plain_split(
            rows,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            rng=rng,
        )

    if args.interleave_template:
        slice_key = args.stratify_key or "a2_slice"

        def _slice(r: dict) -> str:
            meta = r.get("metadata") or {}
            return str(meta.get(slice_key) or meta.get("a1_slice") or meta.get("a2_slice") or "?")

        def _tmpl(r: dict) -> str:
            return meta_template_stem(r.get("metadata") or {})

        train = interleave_a2_samples(train, rng=rng, slice_fn=_slice, template_fn=_tmpl)
        val = interleave_a2_samples(val, rng=random.Random(args.seed + 1), slice_fn=_slice, template_fn=_tmpl)
        test = interleave_a2_samples(test, rng=random.Random(args.seed + 2), slice_fn=_slice, template_fn=_tmpl)
    else:
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, data: list[dict]) -> Path:
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    paths = {
        "train": str(dump("train", train)),
        "val": str(dump("val", val)),
        "test": str(dump("test", test)),
    }
    meta = {
        "seed": args.seed,
        "n_total": len(rows),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": args.test_frac,
        },
        "stratify_key": args.stratify_key or None,
        "interleave_template": bool(args.interleave_template),
        "train_mix": summarize_mix(train) if train else {},
        "paths": paths,
        "source": str(args.input),
    }
    # Prefer curriculum slice key in mix summary when present
    if train and args.stratify_key:
        from collections import Counter

        key = args.stratify_key
        meta["train_mix"]["by_slice"] = dict(
            sorted(Counter(str((r.get("metadata") or {}).get(key) or "?") for r in train).items())
        )
    (args.out_dir / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Split ShareGPT JSONL into train / val / test (by sample, shuffled).

Default: 90% / 5% / 5%. Writes JSONL + a small manifest for notebooks.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import load_jsonl


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
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    # remainder → test (guarantees all samples used)
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]

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
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": args.test_frac,
        },
        "paths": paths,
        "source": str(args.input),
    }
    (args.out_dir / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge synth/real JSONL, optional mix ratio, emit LLaMA-Factory dataset_info + files."""

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
    ap.add_argument("--synth", type=Path, default=ROOT / "data" / "processed" / "synth" / "point_sharegpt.jsonl")
    ap.add_argument("--real", type=Path, default=None)
    ap.add_argument("--real-ratio", type=float, default=0.2, help="Target fraction of real among mixed (0-1)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "llamafactory")
    ap.add_argument("--name", type=str, default="ovisocr2_point")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    synth = load_jsonl(args.synth) if args.synth.exists() else []
    real = load_jsonl(args.real) if args.real and args.real.exists() else []

    if real and args.real_ratio > 0:
        # Mix: keep all synth, subsample or upsample real toward ratio
        # If synth=S, want R/(S+R)=ratio → R = ratio*S/(1-ratio)
        target_r = int(round(args.real_ratio * len(synth) / max(1e-6, 1 - args.real_ratio)))
        if len(real) >= target_r:
            real_use = rng.sample(real, target_r)
        else:
            real_use = list(real)
            while len(real_use) < target_r and real:
                real_use.append(rng.choice(real))
        mixed = synth + real_use
    else:
        mixed = list(synth)

    rng.shuffle(mixed)
    n_val = max(1, int(len(mixed) * args.val_frac)) if mixed else 0
    val = mixed[:n_val]
    train = mixed[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / f"{args.name}_train.json"
    val_path = args.out_dir / f"{args.name}_val.json"
    train_path.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_info = {
        f"{args.name}_train": {
            "file_name": train_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        f"{args.name}_val": {
            "file_name": val_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
    }
    info_path = args.out_dir / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, indent=2), encoding="utf-8")

    # Convenience copy note
    readme = args.out_dir / "README.txt"
    readme.write_text(
        "Copy dataset_info.json entries into your LLaMA-Factory data/dataset_info.json\n"
        f"or set dataset_dir to: {args.out_dir}\n",
        encoding="utf-8",
    )
    print(f"train={len(train)} val={len(val)} → {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge a PEFT LoRA into a local Unsloth vision checkpoint.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python export/merge_unsloth_lora.py \
    --base checkpoints/q1_withreal_merged \
    --adapter checkpoints/grpo_q2_marker/adapter_final \
    --out checkpoints/q1_grpo_merged
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _offline_local(path: Path) -> None:
    if path.is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    args = ap.parse_args()

    base = args.base if args.base.is_dir() else ROOT / args.base
    adapter = args.adapter if args.adapter.is_dir() else ROOT / args.adapter
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if not base.is_dir():
        raise SystemExit(f"missing base {base}")
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"missing adapter {adapter}")

    _offline_local(base)
    from peft import PeftModel
    from unsloth import FastVisionModel

    print(f"loading base {base}", flush=True)
    t0 = time.time()
    model, tokenizer = FastVisionModel.from_pretrained(
        str(base.resolve()),
        load_in_4bit=False,
        dtype=None,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_length,
        local_files_only=True,
    )
    print(f"base loaded in {time.time() - t0:.1f}s", flush=True)
    peft_cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    print(
        f"merging {adapter} peft_type={peft_cfg.get('peft_type')} "
        f"r={peft_cfg.get('r')} alpha={peft_cfg.get('lora_alpha')}",
        flush=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter.resolve()), is_trainable=False)
    if not hasattr(model, "merge_and_unload"):
        raise SystemExit("adapter cannot merge_and_unload")
    model = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    print(f"saving merged → {out}", flush=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    stamp = {
        "base": str(base.resolve()),
        "adapter": str(adapter.resolve()),
        "peft": {k: peft_cfg.get(k) for k in ("peft_type", "r", "lora_alpha", "base_model_name_or_path")},
    }
    (out / "merge_meta.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print(f"done {out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

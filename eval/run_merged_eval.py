#!/usr/bin/env python3
"""Score a *merged* (standalone) VL checkpoint on a POINT JSONL — no LoRA arg.

Needed because a GRPO adapter's PEFT base is the merged SFT checkpoint, so it
must be scored as `<merged SFT> + adapter` (or the already-merged model), never
as `base Instruct + adapter`.

  uv run python eval/run_merged_eval.py \
    --model checkpoints/q1_2b_grpo_merged \
    --data data/splits_stage_q1_withreal/val.jsonl \
    --out-dir checkpoints/q1_2b_grpo_merged/eval_val --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "train"))

from point_ocr.infer import DEFAULT_POINT_MAX_NEW_TOKENS  # noqa: E402
from run_observability import run_final_generate_eval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a merged VL checkpoint (no LoRA)")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="val")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_POINT_MAX_NEW_TOKENS)
    ap.add_argument("--gen-batch-size", type=int, default=1)
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=2048 * 2048,
        help="Resize cap; must match training --max-pixels (default 2048²).",
    )
    ap.add_argument("--min-pixels", type=int, default=448 * 448)
    args = ap.parse_args()

    model_dir = args.model.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model not found: {model_dir}")

    from unsloth import FastVisionModel

    print(f"loading merged model {model_dir}", flush=True)
    model, tokenizer = FastVisionModel.from_pretrained(
        str(model_dir), load_in_4bit=True, use_gradient_checkpointing="unsloth"
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_final_generate_eval(
        model=model,
        tokenizer=tokenizer,
        jsonl_path=args.data.resolve(),
        run_dir=out_dir,
        max_samples=args.max_samples,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
    )
    (out_dir / "metrics" / "eval_meta.json").write_text(
        json.dumps(
            {
                "scored_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "model": str(model_dir),
                "data": str(args.data.resolve()),
                "max_samples": args.max_samples,
                "max_new_tokens": args.max_new_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report.get("overall", {}), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score an existing Stage-A LoRA adapter on a ShareGPT JSONL split (no retrain).

Uses the same Batch-1 decode path as notebook / post-train eval
(stop_strings + format-leak cleanup + format_leak_rate).

Example (full test set, ~250 rows; RTX 4060 laptop ≈ tens of minutes):

  uv run python eval/run_unsloth_adapter_eval.py \\
    --adapter checkpoints/20260912_022155_stage_a/adapter_final \\
    --data data/splits/test.jsonl

Outputs under checkpoints/<run>/eval_test/metrics/ by default
(does not overwrite the original train post-eval under metrics/).
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
    ap = argparse.ArgumentParser(description="Unsloth POINT eval on existing adapter")
    ap.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="LoRA dir (e.g. checkpoints/.../adapter_final)",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "splits" / "test.jsonl",
        help="ShareGPT JSONL to score (default: full test split)",
    )
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write metrics/ (default: <adapter.parent>/metrics)",
    )
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_POINT_MAX_NEW_TOKENS)
    ap.add_argument("--max-image-side", type=int, default=1280)
    ap.add_argument("--hit-threshold", type=float, default=0.85)
    ap.add_argument(
        "--prefix",
        default="test",
        help="Filename prefix: {prefix}_report.json / {prefix}_predictions.jsonl",
    )
    args = ap.parse_args()

    adapter = args.adapter.resolve()
    if not adapter.is_dir():
        raise SystemExit(f"adapter not found: {adapter}")
    data = args.data.resolve()
    if not data.is_file():
        raise SystemExit(f"data not found: {data}")

    prefix = (args.prefix or "test").strip() or "test"
    run_dir = (args.out_dir or (adapter.parent / f"eval_{prefix}")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from unsloth import FastVisionModel

    print(f"loading base={args.model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    print(f"loading adapter={adapter}")
    model = PeftModel.from_pretrained(model, str(adapter))
    FastVisionModel.for_inference(model)

    # Isolated run_dir so we never overwrite the original train post-eval under metrics/.
    report = run_final_generate_eval(
        model=model,
        tokenizer=tokenizer,
        jsonl_path=data,
        run_dir=run_dir,
        max_samples=args.max_samples,
        hit_threshold=args.hit_threshold,
        max_image_side=args.max_image_side,
        max_new_tokens=args.max_new_tokens,
    )

    metrics_dir = run_dir / "metrics"
    final_report = metrics_dir / "final_report.json"
    final_preds = metrics_dir / "final_predictions.jsonl"
    final_plot = metrics_dir / "final_metrics_by_doc_type.png"

    meta = {
        "scored_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "adapter": str(adapter),
        "base_model": args.model,
        "data": str(data),
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "hit_threshold": args.hit_threshold,
        "decode": "batch1_stop_strings_and_format_leak_cleanup",
    }
    report = dict(report)
    report["eval_meta"] = meta

    out_report = metrics_dir / f"{prefix}_report.json"
    out_preds = metrics_dir / f"{prefix}_predictions.jsonl"
    out_plot = metrics_dir / f"{prefix}_metrics_by_doc_type.png"

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if final_preds.exists():
        final_preds.replace(out_preds)
    if final_plot.exists():
        final_plot.replace(out_plot)
    if final_report.exists() and final_report.resolve() != out_report.resolve():
        final_report.unlink(missing_ok=True)

    print(f"wrote {out_report}")
    print(f"wrote {out_preds}")
    if out_plot.exists():
        print(f"wrote {out_plot}")
    print("overall:", json.dumps(report.get("overall", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

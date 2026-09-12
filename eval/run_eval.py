#!/usr/bin/env python3
"""Run POINT inference on held-out manifest and score metrics.

Supports:
  --backend dummy     (echo empty / identity for CI)
  --backend vllm      (OvisOCR2 or finetuned checkpoint via vLLM on CUDA)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from point_ocr.dataset_format import write_jsonl  # noqa: F401
from point_ocr.infer import (
    DEFAULT_POINT_MAX_NEW_TOKENS,
    POINT_STOP_STRINGS,
    strip_format_leak,
)
from point_ocr.metrics import EvalExample, evaluate_examples
from point_ocr.prompts import POINT_PROMPT


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_dummy(rows: list[dict]) -> list[str]:
    # Worst-case baseline simulation: dump a long string for positives
    out = []
    for r in rows:
        if r.get("is_negative"):
            out.append("Full page dump that should not appear on chrome.")
        else:
            # Pretend over-extraction
            out.append((r.get("target") or "") + "\n\n" + "# Extra\n\n" + ("lorem " * 200))
    return out


def predict_vllm(rows: list[dict], model: str, *, max_tokens: int = DEFAULT_POINT_MAX_NEW_TOKENS) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        gdn_prefill_backend="triton",
    )
    prompt = llm.get_tokenizer().apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": POINT_PROMPT}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=list(POINT_STOP_STRINGS),
    )
    images = [Image.open(r["image_path"]).convert("RGB") for r in rows]
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": im},
            "mm_processor_kwargs": {
                "images_kwargs": {"min_pixels": 448 * 448, "max_pixels": 2880 * 2880}
            },
        }
        for im in images
    ]
    outputs = llm.generate(inputs, sampling)
    return [o.outputs[0].text.strip() for o in outputs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "eval" / "heldout" / "manifest.json")
    ap.add_argument("--backend", choices=["dummy", "vllm"], default="dummy")
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "last_report.json")
    ap.add_argument("--pred-out", type=Path, default=None)
    ap.add_argument("--hit-threshold", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_POINT_MAX_NEW_TOKENS)
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    if args.backend == "dummy":
        preds = predict_dummy(rows)
    else:
        preds = predict_vllm(rows, args.model, max_tokens=args.max_tokens)

    examples = []
    pred_rows = []
    for r, p in zip(rows, preds):
        cleaned = strip_format_leak(p)
        examples.append(
            EvalExample(
                sample_id=r["sample_id"],
                prediction=cleaned.cleaned,
                target=r.get("target", ""),
                is_negative=bool(r.get("is_negative")),
                raw_prediction=cleaned.raw,
            )
        )
        pred_rows.append(
            {
                "sample_id": r["sample_id"],
                "prediction": cleaned.cleaned,
                "prediction_raw": cleaned.raw,
                "format_leak": cleaned.format_leak,
                "target": r.get("target", ""),
                "is_negative": bool(r.get("is_negative")),
            }
        )

    report = evaluate_examples(examples, hit_threshold=args.hit_threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if args.pred_out:
        args.pred_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pred_out.open("w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report.to_dict(), indent=2))
    print(
        "Go/No-Go: finetuned must beat zero-shot OvisOCR2+POINT on "
        "block_hit_rate ↑ and over_extraction_rate ↓."
    )


if __name__ == "__main__":
    main()

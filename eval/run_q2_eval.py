#!/usr/bin/env python3
"""Score an existing Q2 (OCR + EN→ZH) prediction JSONL — no GPU, no retrain.

Reads the ``*_predictions.jsonl`` written by
``train/run_observability.run_final_generate_eval`` (post-train eval) or by
``eval/run_unsloth_adapter_eval.py`` and writes a Q2-specific report:

  checkpoints/<run>/metrics/q2_report.json
  checkpoints/<run>/metrics/q2_report.md

Example::

  uv run python eval/run_q2_eval.py \
    --predictions checkpoints/q2_ocr_mt/metrics/final_predictions.jsonl \
    --split data/splits_stage_q2_ocr_mt/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.q2_metrics import (  # noqa: E402
    Q2Example,
    evaluate_q2,
    evaluate_q2_by_bucket,
    report_to_markdown,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metadata_by_sample(split: Path | None) -> dict[str, dict[str, Any]]:
    if split is None or not split.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(split):
        meta = row.get("metadata") or {}
        sid = meta.get("sample_id")
        if sid is not None:
            out[str(sid)] = meta
    return out


def _template_from_sample(sample_id: str, meta: dict[str, Any]) -> str:
    if meta.get("template_stem"):
        return str(meta["template_stem"])
    # core_inner__14_magazine_3col__1131:mag_p3:p3 -> 14_magazine_3col
    page = str(meta.get("page_id") or sample_id.split(":", 1)[0])
    parts = page.split("__")
    return parts[1] if len(parts) > 1 else page


def main() -> None:
    ap = argparse.ArgumentParser(description="Q2 OCR+MT metrics on existing predictions")
    ap.add_argument("--predictions", type=Path, required=True, help="*_predictions.jsonl")
    ap.add_argument("--split", type=Path, default=None, help="source split JSONL for metadata buckets")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: predictions parent dir")
    ap.add_argument("--source-threshold", type=float, default=0.85)
    ap.add_argument("--translation-threshold", type=float, default=0.60)
    ap.add_argument("--prefix", default="q2")
    args = ap.parse_args()

    pred_path = args.predictions.resolve()
    if not pred_path.is_file():
        raise SystemExit(f"predictions not found: {pred_path}")
    preds = load_jsonl(pred_path)
    meta_by_sample = metadata_by_sample(args.split)

    examples: list[Q2Example] = []
    pools: list[str] = []
    templates: list[str] = []
    chromes: list[str] = []
    regions: list[str] = []
    scenes: list[str] = []
    for row in preds:
        sid = str(row.get("sample_id") or len(examples))
        meta = meta_by_sample.get(sid, {})
        gt = row.get("target", "")
        pred = row.get("prediction", "")
        is_neg = bool(row.get("is_negative")) or not str(gt).strip()
        examples.append(
            Q2Example(
                sample_id=sid,
                prediction=str(pred),
                target=str(gt),
                is_negative=is_neg,
                raw_prediction=row.get("prediction_raw"),
            )
        )
        pools.append(str(meta.get("pool_id") or row.get("doc_type") or "unknown"))
        templates.append(_template_from_sample(sid, meta))
        chromes.append(str(meta.get("chrome_family") or "unknown"))
        regions.append(str(meta.get("region") or ("neg" if is_neg else "pos")))
        scenes.append(str(row.get("grpo_scene") or meta.get("grpo_scene") or "regular"))

    overall = evaluate_q2(
        examples,
        source_threshold=args.source_threshold,
        translation_threshold=args.translation_threshold,
    )
    by_pool = evaluate_q2_by_bucket(examples, pools, source_threshold=args.source_threshold, translation_threshold=args.translation_threshold)
    by_template = evaluate_q2_by_bucket(examples, templates, source_threshold=args.source_threshold, translation_threshold=args.translation_threshold)
    by_chrome = evaluate_q2_by_bucket(examples, chromes, source_threshold=args.source_threshold, translation_threshold=args.translation_threshold)
    by_region = evaluate_q2_by_bucket(examples, regions, source_threshold=args.source_threshold, translation_threshold=args.translation_threshold)
    by_scene = evaluate_q2_by_bucket(examples, scenes, source_threshold=args.source_threshold, translation_threshold=args.translation_threshold)

    report: dict[str, Any] = {
        "predictions": str(pred_path),
        "split": str(args.split.resolve()) if args.split else None,
        "n_rows": len(preds),
        "by_pool_counts": dict(Counter(pools)),
        "overall": overall.to_dict(),
        "by_pool": {k: v.to_dict() for k, v in by_pool.items()},
        "by_template": {k: v.to_dict() for k, v in by_template.items()},
        "by_chrome": {k: v.to_dict() for k, v in by_chrome.items()},
        "by_region": {k: v.to_dict() for k, v in by_region.items()},
        "by_scene": {k: v.to_dict() for k, v in by_scene.items()},
    }
    # Markdown uses the pool breakdown as its bucket table.
    md = report_to_markdown({"overall": overall.to_dict(), "by_bucket": {k: v.to_dict() for k, v in by_pool.items()}})

    out_dir = (args.out_dir or pred_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (args.prefix or "q2").strip() or "q2"
    json_path = out_dir / f"{prefix}_report.json"
    md_path = out_dir / f"{prefix}_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(json.dumps(overall.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

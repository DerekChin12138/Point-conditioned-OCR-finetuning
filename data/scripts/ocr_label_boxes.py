#!/usr/bin/env python3
"""Batch OCR for human bbox annotations (label workbench).

Reads ``data/label_out/*.json``. Boxes share a ``unit_id`` when they are
fragments of one semantic block (multi-frag). For each unit:

- any fragment with non-empty text / ``manual`` → keep that text for the whole
  unit (skip OCR on all frags of the unit)
- else crop each frag, OCR with OvisOCR2, join in reading order (top→bottom,
  left→right) with blank lines

Writes enriched JSON under ``data/label_out_ocr/`` including a ``units`` list.

  uv run python data/scripts/ocr_label_boxes.py
  uv run python data/scripts/ocr_label_boxes.py \\
    --label-dir data/label_out --out data/label_out_ocr \\
    --model ATH-MaaS/OvisOCR2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.image_resize import resize_for_ovis  # noqa: E402
from point_ocr.infer import generate_point_text  # noqa: E402
from point_ocr.prompts import PAGE_PROMPT  # noqa: E402


def _load_model(model_id: str):
    import torch
    from unsloth import FastVisionModel

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    print(f"loading {model_id} dtype={dtype}", flush=True)
    model, tok = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=False,
        dtype=dtype,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    print("ready", flush=True)
    return model, tok


def _ocr_crop(model, tok, crop: Image.Image, *, max_new_tokens: int) -> str:
    vis = resize_for_ovis(crop.convert("RGB"))
    out = generate_point_text(
        model,
        tok,
        vis,
        PAGE_PROMPT,
        clean=True,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    return str(out.cleaned) if hasattr(out, "cleaned") else str(out)


def _is_manual(box: dict[str, Any]) -> bool:
    if box.get("manual") is True:
        return True
    return bool(str(box.get("text") or "").strip())


def _clamp_bbox(bbox: list[Any], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = [int(float(v)) for v in bbox]
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    y1, y2 = sorted((max(0, y1), min(h, y2)))
    return [x1, y1, x2, y2]


def _reading_key(bbox: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (y1, x1)


def process_one(
    ann: dict[str, Any],
    *,
    model,
    tok,
    max_new_tokens: int,
) -> dict[str, Any]:
    src = Path(str(ann.get("source_path") or ""))
    if not src.is_file():
        raise FileNotFoundError(f"missing source_path: {src}")
    img = Image.open(src).convert("RGB")
    w, h = img.size

    by_unit: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for i, box in enumerate(ann.get("boxes") or []):
        bbox = box.get("bbox") or []
        if len(bbox) != 4:
            raise ValueError(f"box {i}: bad bbox {bbox}")
        unit_id = str(box.get("unit_id") or f"u{i + 1}").strip() or f"u{i + 1}"
        row = {
            "bbox": _clamp_bbox(bbox, w, h),
            "text": str(box.get("text") or ""),
            "manual": _is_manual(box),
            "unit_id": unit_id,
        }
        by_unit[unit_id].append((i, row))

    boxes_out: list[dict[str, Any] | None] = [None] * len(ann.get("boxes") or [])
    units_out: list[dict[str, Any]] = []
    n_ocr = 0
    n_skip = 0

    for unit_id, members in by_unit.items():
        members_sorted = sorted(members, key=lambda t: _reading_key(t[1]["bbox"]))
        manual_texts = [
            m["text"].strip()
            for _, m in members_sorted
            if m["manual"] and str(m.get("text") or "").strip()
        ]
        frag_bboxes = [m["bbox"] for _, m in members_sorted]
        kind = "multi_frag" if len(members_sorted) > 1 else "single"

        if manual_texts:
            # Prefer first non-empty manual; if several, join in order.
            assembled = "\n\n".join(manual_texts)
            for _, m in members_sorted:
                m["text"] = assembled
                m["manual"] = True
                m["ocr_skipped"] = True
                m["unit_text"] = assembled
            n_skip += 1
            units_out.append(
                {
                    "unit_id": unit_id,
                    "kind": kind,
                    "frag_bboxes": frag_bboxes,
                    "text": assembled,
                    "manual": True,
                    "ocr_skipped": True,
                }
            )
        else:
            frag_texts: list[str] = []
            for _, m in members_sorted:
                x1, y1, x2, y2 = m["bbox"]
                if x2 - x1 < 2 or y2 - y1 < 2:
                    raise ValueError(f"unit {unit_id}: crop too small {m['bbox']}")
                crop = img.crop((x1, y1, x2, y2))
                t = _ocr_crop(model, tok, crop, max_new_tokens=max_new_tokens).strip()
                frag_texts.append(t)
                m["frag_text"] = t
                m["ocr_skipped"] = False
                m["ocr_model"] = "OvisOCR2"
                m["manual"] = False
            assembled = "\n\n".join(t for t in frag_texts if t)
            for _, m in members_sorted:
                m["text"] = assembled
                m["unit_text"] = assembled
            n_ocr += 1
            units_out.append(
                {
                    "unit_id": unit_id,
                    "kind": kind,
                    "frag_bboxes": frag_bboxes,
                    "frag_texts": frag_texts,
                    "text": assembled,
                    "manual": False,
                    "ocr_skipped": False,
                    "ocr_model": "OvisOCR2",
                }
            )

        for orig_i, m in members:
            boxes_out[orig_i] = m

    out = dict(ann)
    out["boxes"] = [b for b in boxes_out if b is not None]
    out["units"] = units_out
    out["n_boxes"] = len(out["boxes"])
    out["n_units"] = len(units_out)
    out["n_manual_units"] = n_skip
    out["n_ocr_units"] = n_ocr
    out["ocr_at"] = datetime.now(timezone.utc).isoformat()
    out["image_wh"] = list(ann.get("image_wh") or [w, h])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-dir", type=Path, default=ROOT / "data/label_out")
    ap.add_argument("--out", type=Path, default=ROOT / "data/label_out_ocr")
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="Max annotation files; 0=all")
    args = ap.parse_args()

    files = sorted(args.label_dir.glob("*.json"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"no *.json under {args.label_dir}", flush=True)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    model, tok = _load_model(args.model)
    summary: list[dict[str, Any]] = []
    for fp in files:
        ann = json.loads(fp.read_text(encoding="utf-8"))
        print(f"ocr {fp.name} …", flush=True)
        try:
            filled = process_one(
                ann, model=model, tok=tok, max_new_tokens=args.max_new_tokens
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {type(e).__name__}: {e}", flush=True)
            summary.append({"file": fp.name, "ok": False, "error": str(e)})
            continue
        dst = args.out / fp.name
        dst.write_text(json.dumps(filled, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  ok units={filled['n_units']} manual={filled['n_manual_units']} "
            f"ocr={filled['n_ocr_units']} → {dst}",
            flush=True,
        )
        summary.append(
            {
                "file": fp.name,
                "ok": True,
                "n_units": filled["n_units"],
                "n_manual_units": filled["n_manual_units"],
                "n_ocr_units": filled["n_ocr_units"],
                "out": str(dst),
            }
        )
    meta = args.out / "ocr_meta.json"
    meta.write_text(
        json.dumps(
            {
                "model": args.model,
                "label_dir": str(args.label_dir),
                "out": str(args.out),
                "n_files": len(files),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {meta}", flush=True)


if __name__ == "__main__":
    main()

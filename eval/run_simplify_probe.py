#!/usr/bin/env python3
"""How much must a real screenshot be simplified before A2-b reads the mark?

Two orthogonal ladders, same 16 nochrome probes, same magenta x45:

  Crop  — shrink the viewport around the point (marker grows after Ovis resize).
  Mask  — keep 2582×1641, paint distractors paper-gray (marker stays tiny).

Mask levels
  chrome  site header / cookie / search / left contents / appearance
  widgets chrome + sidebars, logos, infobox, PDF cards, footer, tools panel
  isolate keep only a 560px window around the point; rest of the full canvas gray

Scoring: hit (sim≥0.85 vs full GT), related (sim≥0.5), empty, chrome-leak.
Tight crops may truncate a wide paragraph; related/in-GT still counts as
"looked at the marked neighborhood."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from build_real_world_probe import BROWSER_CHROME_TOP_PX, CASES  # noqa: E402
from point_ocr.image_resize import resize_for_ovis  # noqa: E402
from point_ocr.infer import generate_point_text  # noqa: E402
from point_ocr.marker import MARKER_SPEC, draw_crosshair  # noqa: E402
from point_ocr.metrics import edit_similarity, is_empty_pred, normalize_text  # noqa: E402
from point_ocr.prompts import POINT_PROMPT  # noqa: E402

HIT = 0.85
RELATED = 0.5
PAPER = (248, 248, 248)
ISOLATE_SIDE = 560
CROP_SIDES = (1100, 720, 480)
KEEP_MARKER_R = 40

# nochrome coords. Boxes are (x0,y0,x1,y1) inclusive-exclusive.
def _page_masks() -> dict[str, dict[str, list[tuple[int, int, int, int]]]]:
    W, H = 2582, 1641
    sb = (W - 48, 0, W, H)  # window scrollbar
    return {
        "0097fb7657aa9131bc2b52cf2745d0fe": {
            "chrome": [(0, 0, W, 90), sb],
            "widgets": [
                (0, 0, W, 90),
                (1780, 90, W, 1180),  # Access Paper rail
                (60, 980, 1780, 1470),  # bibliographic tools
                (0, 1470, W, H),  # funder logos
                sb,
            ],
        },
        "07cc280d49af04b4003229e21e1fcf44": {
            "chrome": [(0, 0, W, 148), sb],
            "widgets": [
                (0, 0, W, 148),  # cookie
                (1580, 160, 2520, 1180),  # journal card
                (2360, 980, W, 1380),  # PDF floater
                sb,
            ],
        },
        "64b7841915a3fc3c754e5f44fd161c41": {
            "chrome": [(0, 0, W, 40), (40, 40, 2000, 280), sb],  # leftover header + blurb + search
            "widgets": [(0, 0, W, 40), (40, 40, 2000, 280), sb],
        },
        "68a0a45aae3e64dd37a772fed83d90c7": {
            "chrome": [(0, 0, W, 230), sb],  # cookie + ABOUT/DOWNLOAD bar
            "widgets": [(0, 0, W, 230), (1500, 230, W, H), sb],  # IMS column
        },
        "7c75e118ee192087b0b2581c60ffaad3": {
            "chrome": [
                (0, 0, W, 125),  # wiki search / account
                (0, 125, 430, H),  # contents
                (2080, 125, W, 980),  # appearance
                sb,
            ],
            "widgets": [
                (0, 0, W, 125),
                (0, 125, 430, H),
                (1420, 180, 2080, H),  # infobox + image
                (2080, 125, W, 980),
                sb,
            ],
        },
    }


CHROME_NEEDLES = (
    "arxiv q search",
    "search submit donate",
    "free distribution service and an open-access",
    "this website uses cookies",
    "a 322 languages",
    "322 languages",
    "donate create account",
    "subject search and browse",
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gt_prompt(row: dict) -> tuple[str, str]:
    msgs = row.get("messages") or []
    gt = msgs[1].get("content", "") if len(msgs) > 1 else ""
    if not isinstance(gt, str):
        gt = str(gt)
    user_raw = msgs[0].get("content", "") if msgs else ""
    if isinstance(user_raw, str) and user_raw.startswith("<image>"):
        prompt = user_raw[len("<image>") :] or POINT_PROMPT
    elif isinstance(user_raw, str) and user_raw.strip():
        prompt = user_raw
    else:
        prompt = POINT_PROMPT
    return gt, prompt


def crop_around(im: Image.Image, x: float, y: float, side: int) -> tuple[Image.Image, float, float]:
    w, h = im.size
    side = min(side, w, h)
    x0 = int(round(x - side / 2))
    y0 = int(round(y - side / 2))
    x0 = max(0, min(w - side, x0))
    y0 = max(0, min(h - side, y0))
    box = (x0, y0, x0 + side, y0 + side)
    return im.crop(box), x - x0, y - y0


def fill_boxes(im: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    out = im.copy()
    dr = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in boxes:
        dr.rectangle([x0, y0, x1 - 1, y1 - 1], fill=PAPER)
    return out


def isolate(im: Image.Image, x: float, y: float, side: int) -> Image.Image:
    w, h = im.size
    cx, cy = int(round(x)), int(round(y))
    half = side // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    out = Image.new("RGB", (w, h), PAPER)
    out.paste(im.crop((x0, y0, x1, y1)), (x0, y0))
    return out


def is_chrome_dump(pred: str) -> bool:
    n = normalize_text(pred)
    if not n:
        return False
    return any(s in n for s in CHROME_NEEDLES)


def pred_in_gt(pred: str, gt: str) -> bool:
    p, g = normalize_text(pred), normalize_text(gt)
    if not p or not g or len(p) < 12:
        return False
    return p in g


def score(pred: str, gt: str) -> dict[str, Any]:
    sim = edit_similarity(pred, gt)
    empty_p = is_empty_pred(pred)
    empty_g = is_empty_pred(gt)
    return {
        "sim": round(sim, 4),
        "hit": bool(sim >= HIT),
        "related": bool(sim >= RELATED),
        "pred_empty": empty_p,
        "gt_empty": empty_g,
        "chrome_dump": is_chrome_dump(pred) and not empty_g,
        "in_gt": pred_in_gt(pred, gt),
    }


def _clip(s: str, n: int = 100) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def load_model(adapter: Path, model_id: str):
    import torch
    from peft import PeftModel
    from unsloth import FastVisionModel

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"loading base={model_id} dtype={dtype}", flush=True)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id, load_in_4bit=False, dtype=dtype, use_gradient_checkpointing="unsloth"
    )
    print(f"loading adapter={adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    FastVisionModel.for_inference(model)
    return model, tokenizer


def infer(model, tokenizer, image: Image.Image, prompt: str, point) -> str:
    img = resize_for_ovis(image.convert("RGB"))
    out = generate_point_text(
        model, tokenizer, img, prompt if prompt.strip() else None, point=point, clean=True
    )
    return str(out.cleaned) if hasattr(out, "cleaned") else str(out)


def treatments_for(stem: str, im: Image.Image, x: float, y: float) -> list[tuple[str, str, Image.Image, float, float]]:
    masks = _page_masks()[stem]
    out: list[tuple[str, str, Image.Image, float, float]] = [
        ("full", "crop", im, x, y),
    ]
    for side in CROP_SIDES:
        cim, cx, cy = crop_around(im, x, y, side)
        out.append((f"crop_{side}", "crop", cim, cx, cy))
    out.append(("mask_chrome", "mask", fill_boxes(im, masks["chrome"]), x, y))
    out.append(("mask_widgets", "mask", fill_boxes(im, masks["widgets"]), x, y))
    out.append(("isolate_560", "mask", isolate(im, x, y, ISOLATE_SIDE), x, y))
    return out


def stem_of(page_id: str) -> str:
    return page_id.removesuffix("__nochrome")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=ROOT / "checkpoints/20260916_092649_stage_a/adapter_final")
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data/real_world_sample/point_sharegpt_nochrome.jsonl",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/processed/simplify_probe")
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()
    stamp_dir = out_dir / "stamped"
    stamp_dir.mkdir(parents=True, exist_ok=True)

    rows = {r["metadata"]["sample_id"]: r for r in load_jsonl(args.jsonl.resolve())}
    model, tokenizer = load_model(args.adapter.resolve(), args.model)

    records: list[dict[str, Any]] = []
    t0 = time.time()
    n_total = len(rows) * (1 + len(CROP_SIDES) + 3)
    i = 0
    for stem, _x, _y, sid, kind, _target in CASES:
        key = f"{sid}__nochrome"
        row = rows.get(key)
        if row is None:
            print(f"[skip] {key}", flush=True)
            continue
        meta = row["metadata"]
        x, y = float(meta["point"][0]), float(meta["point"][1])
        gt, prompt = _gt_prompt(row)
        src = ROOT / "data/real_world_sample/nochrome" / f"{stem}.png"
        base = Image.open(src).convert("RGB")
        for vid, family, page, px, py in treatments_for(stem, base, x, y):
            i += 1
            stamped = draw_crosshair(page, px, py, MARKER_SPEC)
            jpg = stamp_dir / f"{key}__{vid}.jpg"
            stamped.save(jpg, format="JPEG", quality=88)
            print(f"[{i}/{n_total}] {key} {vid} {stamped.size}", flush=True)
            pred = infer(model, tokenizer, stamped, prompt, [px, py])
            sc = score(pred, gt)
            records.append(
                {
                    "sample_id": key,
                    "stem": stem,
                    "kind": str(meta.get("pool_id") or kind),
                    "variant": vid,
                    "family": family,
                    "image_wh": list(stamped.size),
                    "gt": gt,
                    "pred": pred,
                    "gt_clip": _clip(gt),
                    "pred_clip": _clip(pred),
                    **sc,
                }
            )

    elapsed = time.time() - t0

    def agg(sub: list[dict]) -> dict[str, Any]:
        if not sub:
            return {"n": 0}
        pos = [r for r in sub if not r["gt_empty"]]
        neg = [r for r in sub if r["gt_empty"]]
        def rate(xs, key):
            return round(sum(bool(x[key]) for x in xs) / len(xs), 4) if xs else None
        return {
            "n": len(sub),
            "hit": rate(sub, "hit"),
            "related": rate(sub, "related"),
            "empty_pred": rate(sub, "pred_empty"),
            "chrome_dump": rate(sub, "chrome_dump"),
            "pos_n": len(pos),
            "pos_hit": rate(pos, "hit"),
            "pos_related": rate(pos, "related"),
            "pos_in_gt": rate(pos, "in_gt"),
            "pos_empty": rate(pos, "pred_empty"),
            "pos_chrome": rate(pos, "chrome_dump"),
            "neg_n": len(neg),
            "neg_empty": rate(neg, "pred_empty"),
        }

    variants = ["full", "crop_1100", "crop_720", "crop_480", "mask_chrome", "mask_widgets", "isolate_560"]
    by_variant = {v: agg([r for r in records if r["variant"] == v]) for v in variants}

    # first level where each positive hits
    first_hit: list[dict[str, Any]] = []
    for sid in sorted({r["sample_id"] for r in records}):
        rs = {r["variant"]: r for r in records if r["sample_id"] == sid}
        if rs["full"]["gt_empty"]:
            continue
        unlocked = next((v for v in variants if rs[v]["hit"]), None)
        related_at = next((v for v in variants if rs[v]["related"] or rs[v]["in_gt"]), None)
        first_hit.append(
            {
                "sample_id": sid,
                "kind": rs["full"]["kind"],
                "gt_clip": rs["full"]["gt_clip"],
                "first_hit": unlocked,
                "first_related": related_at,
                "by_variant": {
                    v: {
                        "hit": rs[v]["hit"],
                        "related": rs[v]["related"],
                        "empty": rs[v]["pred_empty"],
                        "chrome": rs[v]["chrome_dump"],
                        "in_gt": rs[v]["in_gt"],
                        "sim": rs[v]["sim"],
                        "pred": rs[v]["pred_clip"],
                    }
                    for v in variants
                },
            }
        )

    summary = {
        "scored_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "adapter": str(args.adapter.resolve()),
        "elapsed_sec": round(elapsed, 1),
        "n_inferences": len(records),
        "by_variant": by_variant,
        "positives": first_hit,
        "negatives": [
            {
                "sample_id": sid,
                "kind": next(r["kind"] for r in records if r["sample_id"] == sid),
                "by_variant": {
                    v: {
                        "empty": r["pred_empty"],
                        "hit": r["hit"],
                        "pred": r["pred_clip"],
                    }
                    for v in variants
                    for r in records
                    if r["sample_id"] == sid and r["variant"] == v
                },
            }
            for sid in sorted({r["sample_id"] for r in records if r["gt_empty"]})
        ],
    }
    (out_dir / "simplify_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    out = out_dir / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(json.dumps(by_variant, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

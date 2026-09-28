#!/usr/bin/env python3
"""Marker cue probes for the latest Stage-A adapter (A2-b / adapter_final).

1. Synth val subsample: restamp unmarked renders with recolored X / reshaped
   magenta markers (plus a no-marker control), then greedy-decode.
2. Real nochrome screenshots: grayscale page + magenta x45 vs color page +
   magenta x45.

Outputs a JSON summary consumed by the experiment canvas.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.image_resize import resize_for_ovis  # noqa: E402
from point_ocr.infer import generate_point_text  # noqa: E402
from point_ocr.marker import MARKER_SPEC, draw_crosshair  # noqa: E402
from point_ocr.metrics import edit_similarity, is_empty_pred, normalize_text  # noqa: E402
from point_ocr.prompts import POINT_PROMPT  # noqa: E402

HIT_THRESHOLD = 0.85
SEED = 42
SYNTH_QUOTAS = {
    "core_inner": 4,
    "multi_frag": 4,
    "semantic_group": 4,
    "empty_boundary": 4,
    "empty_clear": 4,
}


def _rgba(rgb: tuple[int, int, int], a: int = 255) -> tuple[int, int, int, int]:
    return (rgb[0], rgb[1], rgb[2], a)


def spec_colored(rgb: tuple[int, int, int], rotation_deg: float = 45.0):
    c = _rgba(rgb)
    return replace(
        MARKER_SPEC,
        cross_color=c,
        center_dot_color=c,
        rotation_deg=rotation_deg,
    )


def draw_disk(
    image: Image.Image,
    x: float,
    y: float,
    *,
    fill: tuple[int, int, int, int] = (255, 0, 255, 255),
    outline: tuple[int, int, int, int] = (255, 255, 255, 255),
    radius: int = 22,
    outline_width: int = 3,
) -> Image.Image:
    base = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = int(round(x)), int(round(y))
    w, h = base.size
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    box = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(box, fill=fill)
    draw.ellipse(box, outline=outline, width=outline_width)
    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_square(
    image: Image.Image,
    x: float,
    y: float,
    *,
    fill: tuple[int, int, int, int] = (255, 0, 255, 255),
    outline: tuple[int, int, int, int] = (255, 255, 255, 255),
    half: int = 18,
    outline_width: int = 3,
) -> Image.Image:
    base = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = int(round(x)), int(round(y))
    w, h = base.size
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    box = [cx - half, cy - half, cx + half, cy + half]
    draw.rectangle(box, fill=fill, outline=outline, width=outline_width)
    return Image.alpha_composite(base, overlay).convert("RGB")


# (variant_id, family, label, drawer)
# family: baseline | color | shape
VARIANTS: list[tuple[str, str, str, Callable[[Image.Image, float, float], Image.Image]]] = [
    (
        "x45_magenta",
        "baseline",
        "X 45° magenta (train)",
        lambda im, x, y: draw_crosshair(im, x, y, MARKER_SPEC),
    ),
    (
        "x45_cyan",
        "color",
        "X 45° cyan",
        lambda im, x, y: draw_crosshair(im, x, y, spec_colored((0, 220, 255))),
    ),
    (
        "x45_lime",
        "color",
        "X 45° lime",
        lambda im, x, y: draw_crosshair(im, x, y, spec_colored((50, 205, 50))),
    ),
    (
        "x45_yellow",
        "color",
        "X 45° yellow",
        lambda im, x, y: draw_crosshair(im, x, y, spec_colored((255, 210, 0))),
    ),
    (
        "x45_black",
        "color",
        "X 45° black",
        lambda im, x, y: draw_crosshair(im, x, y, spec_colored((20, 20, 24))),
    ),
    (
        "plus_magenta",
        "shape",
        "+ 0° magenta",
        lambda im, x, y: draw_crosshair(im, x, y, spec_colored((255, 0, 255), rotation_deg=0.0)),
    ),
    (
        "disk_magenta",
        "shape",
        "disk magenta",
        lambda im, x, y: draw_disk(im, x, y),
    ),
    (
        "square_magenta",
        "shape",
        "square magenta",
        lambda im, x, y: draw_square(im, x, y),
    ),
    (
        "none",
        "shape",
        "no marker",
        lambda im, x, y: im.convert("RGB"),
    ),
]


def _gt_and_prompt(row: dict) -> tuple[str, str]:
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


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_synth(val_path: Path, rng_seed: int = SEED) -> list[dict]:
    import random

    rng = random.Random(rng_seed)
    rows = _load_jsonl(val_path)
    by_pool: dict[str, list[dict]] = {}
    for row in rows:
        meta = row.get("metadata") or {}
        pool = str(meta.get("pool_id") or "unknown")
        page_id = meta.get("page_id")
        point = meta.get("point")
        if not page_id or not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        render = ROOT / "data" / "pools" / pool / "renders" / f"{page_id}.png"
        if not render.is_file():
            continue
        by_pool.setdefault(pool, []).append(row)
    picked: list[dict] = []
    for pool, quota in SYNTH_QUOTAS.items():
        bucket = list(by_pool.get(pool, []))
        rng.shuffle(bucket)
        if len(bucket) < quota:
            raise SystemExit(f"pool {pool}: only {len(bucket)} usable rows, need {quota}")
        picked.extend(bucket[:quota])
    return picked


def _score(pred: str, gt: str) -> dict[str, Any]:
    sim = edit_similarity(pred, gt)
    empty_p = is_empty_pred(pred)
    empty_g = is_empty_pred(gt)
    hit = bool(sim >= HIT_THRESHOLD)
    return {
        "sim": round(sim, 4),
        "hit": hit,
        "pred_empty": empty_p,
        "gt_empty": empty_g,
    }


def _clip(s: str, n: int = 160) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def load_model(adapter: Path, model_id: str, load_in_4bit: bool):
    import torch
    from peft import PeftModel
    from unsloth import FastVisionModel

    load_kwargs: dict[str, Any] = dict(
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    if not load_in_4bit:
        load_kwargs["dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        )
    print(f"loading base={model_id} 4bit={load_in_4bit} dtype={load_kwargs.get('dtype')}", flush=True)
    model, tokenizer = FastVisionModel.from_pretrained(model_id, **load_kwargs)
    print(f"loading adapter={adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    FastVisionModel.for_inference(model)
    return model, tokenizer


def infer_one(model, tokenizer, image: Image.Image, prompt: str, point) -> str:
    img = resize_for_ovis(image.convert("RGB"))
    out = generate_point_text(
        model,
        tokenizer,
        img,
        prompt if prompt.strip() else None,
        point=point if isinstance(point, (list, tuple)) else None,
        clean=True,
    )
    if hasattr(out, "cleaned"):
        return str(out.cleaned)
    return str(out)


def run_synth(model, tokenizer, samples: list[dict], out_dir: Path) -> dict[str, Any]:
    stamp_dir = out_dir / "stamped_synth"
    stamp_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    n = len(samples) * len(VARIANTS)
    i = 0
    t0 = time.time()
    for row in samples:
        meta = row["metadata"]
        pool = str(meta["pool_id"])
        page_id = str(meta["page_id"])
        sid = str(meta.get("sample_id") or page_id)
        point = meta["point"]
        x, y = float(point[0]), float(point[1])
        gt, prompt = _gt_and_prompt(row)
        render = ROOT / "data" / "pools" / pool / "renders" / f"{page_id}.png"
        base = Image.open(render).convert("RGB")
        mw, mh = int(meta.get("image_w") or 0), int(meta.get("image_h") or 0)
        if mw and mh and (base.size != (mw, mh)):
            # Point is in marked-image pixels; scale if render disagrees.
            sx = base.size[0] / mw
            sy = base.size[1] / mh
            x, y = x * sx, y * sy
        for vid, family, label, drawer in VARIANTS:
            i += 1
            stamped = drawer(base, x, y)
            jpg = stamp_dir / f"{sid}__{vid}.jpg"
            if not jpg.exists():
                stamped.save(jpg, format="JPEG", quality=92)
            print(f"[{i}/{n}] synth {sid} {vid}", flush=True)
            pred = infer_one(model, tokenizer, stamped, prompt, [x, y])
            sc = _score(pred, gt)
            records.append(
                {
                    "split": "synth_val",
                    "sample_id": sid,
                    "pool_id": pool,
                    "template_stem": meta.get("template_stem"),
                    "variant": vid,
                    "family": family,
                    "label": label,
                    "gt": gt,
                    "pred": pred,
                    "gt_clip": _clip(gt, 120),
                    "pred_clip": _clip(pred, 120),
                    **sc,
                }
            )
    elapsed = time.time() - t0
    by_sid: dict[str, dict[str, str]] = {}
    for rec in records:
        by_sid.setdefault(rec["sample_id"], {})[rec["variant"]] = rec["pred"]
    for rec in records:
        base_pred = by_sid[rec["sample_id"]].get("x45_magenta", "")
        rec["agree_baseline"] = normalize_text(rec["pred"]) == normalize_text(base_pred)

    def _agg(subset: list[dict]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        pos = [r for r in subset if not r["gt_empty"]]
        neg = [r for r in subset if r["gt_empty"]]
        return {
            "n": len(subset),
            "hit_rate": round(sum(r["hit"] for r in subset) / len(subset), 4),
            "mean_sim": round(sum(r["sim"] for r in subset) / len(subset), 4),
            "empty_pred_rate": round(sum(r["pred_empty"] for r in subset) / len(subset), 4),
            "pos_n": len(pos),
            "pos_hit_rate": round(sum(r["hit"] for r in pos) / len(pos), 4) if pos else None,
            "neg_n": len(neg),
            "neg_empty_rate": round(sum(r["pred_empty"] for r in neg) / len(neg), 4) if neg else None,
            "agree_baseline": round(sum(r["agree_baseline"] for r in subset) / len(subset), 4),
        }

    by_variant: dict[str, Any] = {}
    for vid, family, label, _ in VARIANTS:
        sub = [r for r in records if r["variant"] == vid]
        by_variant[vid] = {"family": family, "label": label, **_agg(sub)}

    by_pool_variant: dict[str, dict[str, Any]] = {}
    for rec in records:
        by_pool_variant.setdefault(rec["pool_id"], {})
        by_pool_variant[rec["pool_id"]].setdefault(rec["variant"], {"n": 0, "hits": 0})
        by_pool_variant[rec["pool_id"]][rec["variant"]]["n"] += 1
        by_pool_variant[rec["pool_id"]][rec["variant"]]["hits"] += int(rec["hit"])
    pool_table: dict[str, dict[str, float]] = {}
    for pool, variants in by_pool_variant.items():
        pool_table[pool] = {
            vid: round(v["hits"] / v["n"], 4) if v["n"] else 0.0 for vid, v in variants.items()
        }

    (out_dir / "synth_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return {
        "n_samples": len(samples),
        "n_inferences": len(records),
        "elapsed_sec": round(elapsed, 1),
        "quotas": SYNTH_QUOTAS,
        "by_variant": by_variant,
        "hit_by_pool_variant": pool_table,
        "records": records,
    }


def _nochrome_src(page_id: str) -> Path:
    stem = page_id.removesuffix("__nochrome")
    return ROOT / "data" / "real_world_sample" / "nochrome" / f"{stem}.png"


def to_gray_rgb(image: Image.Image) -> Image.Image:
    return image.convert("L").convert("RGB")


def run_real(model, tokenizer, jsonl_path: Path, out_dir: Path) -> dict[str, Any]:
    stamp_dir = out_dir / "stamped_real"
    stamp_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(jsonl_path)
    records: list[dict[str, Any]] = []
    t0 = time.time()
    n = len(rows) * 2
    i = 0
    for row in rows:
        meta = row["metadata"]
        sid = str(meta.get("sample_id"))
        page_id = str(meta.get("page_id"))
        point = meta["point"]
        x, y = float(point[0]), float(point[1])
        gt, prompt = _gt_and_prompt(row)
        src = _nochrome_src(page_id)
        if not src.is_file():
            print(f"[skip] missing {src}", flush=True)
            continue
        color = Image.open(src).convert("RGB")
        gray = to_gray_rgb(color)
        for vid, page, label in (
            ("color_x45", color, "color page + magenta X"),
            ("gray_x45", gray, "grayscale page + magenta X"),
        ):
            i += 1
            stamped = draw_crosshair(page, x, y, MARKER_SPEC)
            jpg = stamp_dir / f"{sid}__{vid}.jpg"
            stamped.save(jpg, format="JPEG", quality=92)
            print(f"[{i}/{n}] real {sid} {vid}", flush=True)
            pred = infer_one(model, tokenizer, stamped, prompt, [x, y])
            sc = _score(pred, gt)
            records.append(
                {
                    "split": "real_nochrome",
                    "sample_id": sid,
                    "pool_id": str(meta.get("pool_id") or ""),
                    "variant": vid,
                    "label": label,
                    "gt": gt,
                    "pred": pred,
                    "gt_clip": _clip(gt, 120),
                    "pred_clip": _clip(pred, 120),
                    **sc,
                }
            )
    elapsed = time.time() - t0
    paired: dict[str, dict[str, dict]] = {}
    for rec in records:
        paired.setdefault(rec["sample_id"], {})[rec["variant"]] = rec
    flips: list[dict[str, Any]] = []
    for sid, d in paired.items():
        a, b = d.get("color_x45"), d.get("gray_x45")
        if not a or not b:
            continue
        flips.append(
            {
                "sample_id": sid,
                "pool_id": a["pool_id"],
                "gt_empty": a["gt_empty"],
                "gt_clip": a["gt_clip"],
                "color_hit": a["hit"],
                "gray_hit": b["hit"],
                "color_empty": a["pred_empty"],
                "gray_empty": b["pred_empty"],
                "color_pred": a["pred_clip"],
                "gray_pred": b["pred_clip"],
                "agree": normalize_text(a["pred"]) == normalize_text(b["pred"]),
                "improved": (not a["hit"]) and b["hit"],
                "regressed": a["hit"] and (not b["hit"]),
            }
        )

    def _agg(vid: str) -> dict[str, Any]:
        sub = [r for r in records if r["variant"] == vid]
        if not sub:
            return {"n": 0}
        pos = [r for r in sub if not r["gt_empty"]]
        neg = [r for r in sub if r["gt_empty"]]
        return {
            "n": len(sub),
            "hit_rate": round(sum(r["hit"] for r in sub) / len(sub), 4),
            "mean_sim": round(sum(r["sim"] for r in sub) / len(sub), 4),
            "empty_pred_rate": round(sum(r["pred_empty"] for r in sub) / len(sub), 4),
            "pos_n": len(pos),
            "pos_hit_rate": round(sum(r["hit"] for r in pos) / len(pos), 4) if pos else None,
            "neg_n": len(neg),
            "neg_empty_rate": round(sum(r["pred_empty"] for r in neg) / len(neg), 4) if neg else None,
        }

    (out_dir / "real_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return {
        "n_samples": len(paired),
        "n_inferences": len(records),
        "elapsed_sec": round(elapsed, 1),
        "by_variant": {
            "color_x45": {"label": "color page + magenta X", **_agg("color_x45")},
            "gray_x45": {"label": "grayscale page + magenta X", **_agg("gray_x45")},
        },
        "n_improved": sum(1 for f in flips if f["improved"]),
        "n_regressed": sum(1 for f in flips if f["regressed"]),
        "n_agree": sum(1 for f in flips if f["agree"]),
        "flips": flips,
        "records": records,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "checkpoints/20260916_092649_stage_a/adapter_final",
    )
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument("--val", type=Path, default=ROOT / "data/splits_stage_a2b/val.jsonl")
    ap.add_argument(
        "--real-jsonl",
        type=Path,
        default=ROOT / "data/real_world_sample/point_sharegpt_nochrome.jsonl",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/processed/marker_probe")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_rows = sample_synth(args.val.resolve())
    print(
        "sampled",
        len(synth_rows),
        [(r["metadata"]["pool_id"], r["metadata"]["sample_id"]) for r in synth_rows],
        flush=True,
    )

    model, tokenizer = load_model(args.adapter.resolve(), args.model, args.load_in_4bit)
    synth = run_synth(model, tokenizer, synth_rows, out_dir)
    real = run_real(model, tokenizer, args.real_jsonl.resolve(), out_dir)

    summary = {
        "scored_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "adapter": str(args.adapter.resolve()),
        "base_model": args.model,
        "load_in_4bit": bool(args.load_in_4bit),
        "hit_threshold": HIT_THRESHOLD,
        "seed": SEED,
        "synth": {k: v for k, v in synth.items() if k != "records"},
        "real": {k: v for k, v in real.items() if k != "records"},
        "synth_examples": [
            {
                "sample_id": r["sample_id"],
                "pool_id": r["pool_id"],
                "variant": r["variant"],
                "hit": r["hit"],
                "agree_baseline": r["agree_baseline"],
                "pred_empty": r["pred_empty"],
                "gt_empty": r["gt_empty"],
                "sim": r["sim"],
                "gt_clip": r["gt_clip"],
                "pred_clip": r["pred_clip"],
            }
            for r in synth["records"]
        ],
        "real_examples": [
            {
                "sample_id": r["sample_id"],
                "pool_id": r["pool_id"],
                "variant": r["variant"],
                "hit": r["hit"],
                "pred_empty": r["pred_empty"],
                "gt_empty": r["gt_empty"],
                "sim": r["sim"],
                "gt_clip": r["gt_clip"],
                "pred_clip": r["pred_clip"],
            }
            for r in real["records"]
        ],
    }
    out = out_dir / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(json.dumps(summary["synth"]["by_variant"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["real"]["by_variant"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

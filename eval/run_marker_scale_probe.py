#!/usr/bin/env python3
"""Scale the train x45 marker on full real screenshots (no crop).

Question: how large must the magenta X be to beat site chrome on 2582px pages?
Also A/B A1 (curriculum_a1) vs A2-b adapters.

Scales are linear on MarkerSpec pixel fields. 1x is the train protocol
(arm ~68px). The page is never cropped.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from build_real_world_probe import CASES  # noqa: E402
from point_ocr.image_resize import resize_for_ovis  # noqa: E402
from point_ocr.infer import generate_point_text  # noqa: E402
from point_ocr.marker import MARKER_SPEC, MarkerSpec, draw_crosshair  # noqa: E402
from point_ocr.metrics import edit_similarity, is_empty_pred, normalize_text  # noqa: E402
from point_ocr.prompts import POINT_PROMPT  # noqa: E402

HIT = 0.85
RELATED = 0.5
SCALES = (1, 2, 3, 4, 6)

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

ADAPTERS = {
    "a1": ROOT / "checkpoints/20260917_002139_curriculum_b1/adapter_final",
    "a2b": ROOT / "checkpoints/20260916_092649_stage_a/adapter_final",
}


def scale_spec(s: float) -> MarkerSpec:
    b = MARKER_SPEC
    return replace(
        b,
        ring_radius_px=max(4, int(round(b.ring_radius_px * s))),
        ring_width_px=max(1, int(round(b.ring_width_px * s))),
        cross_half_length_px=max(6, int(round(b.cross_half_length_px * s))),
        cross_width_px=max(1, int(round(b.cross_width_px * s))),
        cross_outline_width_px=max(1, int(round(b.cross_outline_width_px * s))),
        center_dot_radius_px=max(1, int(round(b.center_dot_radius_px * s))),
        min_visible_extent_px=max(8, int(round(b.min_visible_extent_px * s))),
    )


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gt_prompt(row: dict) -> tuple[str, str]:
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


def clip(s: str, n: int = 100) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def load_model(adapter: Path, model_id: str):
    import torch
    from peft import PeftModel
    from unsloth import FastVisionModel

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
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


def unload(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def agg(sub: list[dict]) -> dict[str, Any]:
    if not sub:
        return {"n": 0}

    def rate(xs, key):
        return round(sum(bool(x[key]) for x in xs) / len(xs), 4) if xs else None

    pos = [r for r in sub if not r["gt_empty"]]
    neg = [r for r in sub if r["gt_empty"]]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data/real_world_sample/point_sharegpt_nochrome.jsonl",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/processed/marker_scale_probe")
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()
    stamp_dir = out_dir / "stamped"
    stamp_dir.mkdir(parents=True, exist_ok=True)

    rows = {r["metadata"]["sample_id"]: r for r in load_jsonl(args.jsonl.resolve())}
    jobs: list[dict[str, Any]] = []
    for stem, _x, _y, sid, kind, _target in CASES:
        key = f"{sid}__nochrome"
        row = rows.get(key)
        if row is None:
            print(f"[skip] {key}", flush=True)
            continue
        meta = row["metadata"]
        x, y = float(meta["point"][0]), float(meta["point"][1])
        gt, prompt = gt_prompt(row)
        src = ROOT / "data/real_world_sample/nochrome" / f"{stem}.png"
        jobs.append(
            {
                "sample_id": key,
                "stem": stem,
                "kind": str(meta.get("pool_id") or kind),
                "x": x,
                "y": y,
                "gt": gt,
                "prompt": prompt,
                "src": src,
            }
        )

    records: list[dict[str, Any]] = []
    t0 = time.time()
    n_total = len(jobs) * len(SCALES) * len(ADAPTERS)
    i = 0
    for model_key, adapter in ADAPTERS.items():
        model, tokenizer = load_model(adapter, args.model)
        for job in jobs:
            base = Image.open(job["src"]).convert("RGB")
            for s in SCALES:
                i += 1
                spec = scale_spec(s)
                vid = f"x{s:g}"
                stamped = draw_crosshair(base, job["x"], job["y"], spec)
                jpg = stamp_dir / f"{model_key}__{job['sample_id']}__{vid}.jpg"
                if not jpg.exists():
                    stamped.save(jpg, format="JPEG", quality=88)
                print(
                    f"[{i}/{n_total}] {model_key} {job['sample_id']} {vid} "
                    f"arm={spec.cross_half_length_px * 2}px",
                    flush=True,
                )
                pred = infer(model, tokenizer, stamped, job["prompt"], [job["x"], job["y"]])
                sc = score(pred, job["gt"])
                records.append(
                    {
                        "model": model_key,
                        "adapter": str(adapter),
                        "sample_id": job["sample_id"],
                        "kind": job["kind"],
                        "scale": s,
                        "variant": vid,
                        "extent_px": spec.cross_half_length_px * 2,
                        "gt": job["gt"],
                        "pred": pred,
                        "gt_clip": clip(job["gt"]),
                        "pred_clip": clip(pred),
                        **sc,
                    }
                )
        unload(model)

    elapsed = time.time() - t0
    by_model: dict[str, Any] = {}
    for mk in ADAPTERS:
        by_model[mk] = {
            f"x{s:g}": agg([r for r in records if r["model"] == mk and r["scale"] == s])
            for s in SCALES
        }

    positives: list[dict[str, Any]] = []
    for mk in ADAPTERS:
        for sid in sorted({r["sample_id"] for r in records if not r["gt_empty"]}):
            rs = {
                r["scale"]: r
                for r in records
                if r["model"] == mk and r["sample_id"] == sid
            }
            first_hit = next((s for s in SCALES if rs[s]["hit"]), None)
            first_rel = next(
                (s for s in SCALES if rs[s]["related"] or rs[s]["in_gt"]), None
            )
            positives.append(
                {
                    "model": mk,
                    "sample_id": sid,
                    "kind": rs[SCALES[0]]["kind"],
                    "gt_clip": rs[SCALES[0]]["gt_clip"],
                    "first_hit": first_hit,
                    "first_related": first_rel,
                    "by_scale": {
                        str(s): {
                            "hit": rs[s]["hit"],
                            "related": rs[s]["related"],
                            "empty": rs[s]["pred_empty"],
                            "chrome": rs[s]["chrome_dump"],
                            "in_gt": rs[s]["in_gt"],
                            "sim": rs[s]["sim"],
                            "pred": rs[s]["pred_clip"],
                        }
                        for s in SCALES
                    },
                }
            )

    summary = {
        "scored_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "n_inferences": len(records),
        "scales": list(SCALES),
        "extent_px": {f"x{s:g}": scale_spec(s).cross_half_length_px * 2 for s in SCALES},
        "adapters": {k: str(v) for k, v in ADAPTERS.items()},
        "by_model": by_model,
        "positives": positives,
        "negatives": [
            {
                "model": mk,
                "sample_id": sid,
                "kind": next(
                    r["kind"]
                    for r in records
                    if r["model"] == mk and r["sample_id"] == sid
                ),
                "by_scale": {
                    str(s): {
                        "empty": r["pred_empty"],
                        "hit": r["hit"],
                        "pred": r["pred_clip"],
                    }
                    for s in SCALES
                    for r in records
                    if r["model"] == mk and r["sample_id"] == sid and r["scale"] == s
                },
            }
            for mk in ADAPTERS
            for sid in sorted({r["sample_id"] for r in records if r["gt_empty"]})
        ],
    }
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    out = out_dir / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(json.dumps(by_model, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Copy real_labeled units into OCR-MT XML with mixed marker sizes.

Latin units: translate with POINT_OCR_LLM_* (capped). CJK units: translation
repeats the Chinese source so the model is not taught Chinese→English.

  uv run python data/scripts/build_real_mt_pool.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from generate_template_content import _load_dotenv, chat_json  # noqa: E402
from point_ocr.dataset_format import load_jsonl, write_jsonl, PointSample  # noqa: E402
from point_ocr.marker import (  # noqa: E402
    CURRENT_MARKER_TAG,
    MARKER_AREA_FRAC_LEGACY,
    draw_crosshair,
    sample_marker_area_frac,
    spec_for_image,
)
from point_ocr.ocr_mt import wrap_point_target  # noqa: E402

SRC_JSONL = ROOT / "data/pools_real/real_labeled/point_sharegpt.jsonl"
OUT = ROOT / "data/pools_real/real_labeled_mt"
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def _is_cjk(text: str) -> bool:
    return len(_CJK.findall(text)) >= max(2, len(_LATIN.findall(text)))


def _assistant(row: dict[str, Any]) -> str:
    msgs = row.get("messages") or []
    if msgs and msgs[-1].get("role") == "assistant":
        return str(msgs[-1].get("content") or "")
    return ""


def _translate_batch(texts: list[str], *, model: str, base_url: str, api_key: str) -> dict[str, str]:
    if not texts:
        return {}
    payload = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user = (
        "Translate each numbered English passage into faithful Simplified Chinese. "
        "Keep meaning, names, and numbers. Return JSON: "
        '{"items":["zh1","zh2",...]} in the same order.\n\n' + payload
    )
    data = chat_json(
        base_url=base_url,
        api_key=api_key,
        model=model,
        user=user,
        temperature=0.2,
        timeout_s=90.0,
    )
    items = data.get("items") or []
    out: dict[str, str] = {}
    for src, zh in zip(texts, items):
        z = str(zh).strip()
        if z:
            out[src] = z
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC_JSONL)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--api-cap", type=int, default=400, help="Max unique Latin strings to translate via API.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    import os

    base_url = os.environ.get("POINT_OCR_LLM_BASE_URL") or ""
    api_key = os.environ.get("POINT_OCR_LLM_API_KEY") or ""
    model = os.environ.get("POINT_OCR_LLM_MODEL") or ""

    rows = load_jsonl(args.src)
    rng = random.Random(args.seed)
    unique_latin: list[str] = []
    seen: set[str] = set()
    for row in rows:
        md = _assistant(row).strip()
        if not md or md in seen:
            continue
        seen.add(md)
        if not _is_cjk(md):
            unique_latin.append(md)
    rng.shuffle(unique_latin)
    to_api = unique_latin[: max(0, args.api_cap)]
    translations: dict[str, str] = {}
    if to_api and base_url and api_key and model:
        print(f"[api] translating {len(to_api)} unique Latin units (cap={args.api_cap})", flush=True)
        batch: list[str] = []
        for t in to_api:
            batch.append(t)
            if len(batch) >= 8:
                translations.update(_translate_batch(batch, model=model, base_url=base_url, api_key=api_key))
                print(f"  api {len(translations)}/{len(to_api)}", flush=True)
                batch = []
        if batch:
            translations.update(_translate_batch(batch, model=model, base_url=base_url, api_key=api_key))
    else:
        print("[api] skipped (missing env or empty latin set)", flush=True)

    marked_dir = args.out / "marked"
    marked_dir.mkdir(parents=True, exist_ok=True)
    samples: list[PointSample] = []
    skipped_latin = 0
    n_cjk = 0
    n_api = 0
    scales = Counter()
    for i, row in enumerate(rows):
        md = _assistant(row).strip()
        meta = dict(row.get("metadata") or {})
        is_neg = bool(meta.get("is_negative")) or not md
        zh = ""
        if not is_neg:
            if _is_cjk(md):
                zh = md
                n_cjk += 1
            elif md in translations:
                zh = translations[md]
                n_api += 1
            else:
                skipped_latin += 1
                # still wrap with empty translation rather than drop the OCR example
                zh = ""
        target = "" if is_neg else wrap_point_target(md, prompt_key="ocr_mt_v1", translation=zh)
        point = meta.get("point") or [0, 0]
        imgs = row.get("images") or []
        src_img = Path(str(imgs[0])) if imgs else None
        # Prefer unmarked render if present next to original marked jpg.
        page_id = str(meta.get("page_id") or "")
        render = ROOT / "data/pools_real/real_labeled/renders" / f"{page_id.split('__', 1)[-1]}.png"
        if not render.is_file() and src_img and src_img.is_file():
            render = src_img
        if not render.is_file():
            continue
        img = Image.open(render).convert("RGB")
        w, h = img.size
        area_frac = sample_marker_area_frac(rng)
        scales[f"{area_frac:.4f}"] += 1
        spec = spec_for_image(w, h, area_frac=area_frac)
        stamped = draw_crosshair(img, float(point[0]), float(point[1]), spec)
        name = f"{meta.get('sample_id', i)}__a{area_frac:.4f}.jpg".replace(":", "_")
        path = marked_dir / name
        stamped.save(path, format="JPEG", quality=92)
        meta.update(
            {
                "prompt_key": "ocr_mt_v1",
                "marker": CURRENT_MARKER_TAG,
                "marker_area_frac": float(area_frac),
                "marker_area_scale": float(area_frac) / MARKER_AREA_FRAC_LEGACY,
                "translation": zh,
                "pool_id": "real_labeled",
            }
        )
        samples.append(
            PointSample(
                sample_id=str(meta.get("sample_id") or f"real_mt_{i}"),
                image_path=str(path.resolve()),
                task="POINT",
                target=target,
                meta=meta,
            )
        )
        img.close()

    n = write_jsonl(args.out / "point_sharegpt.jsonl", samples)
    stats = {
        "n": n,
        "n_cjk_identity": n_cjk,
        "n_api_zh": n_api,
        "skipped_latin": skipped_latin,
        "marker_scales": dict(scales),
        "api_unique": len(translations),
    }
    (args.out / "pool_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()

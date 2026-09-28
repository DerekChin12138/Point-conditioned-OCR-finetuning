#!/usr/bin/env python3
"""Build ``data/pools_real/real_labeled`` from human bbox labels + OvisOCR2 text.

Steps:
  1. Ensure ``data/label_out_ocr/*.json`` exists (runs OCR if missing / --force-ocr).
  2. For each text unit: diamond5 (5×2) on every fragment bbox, stamp X,
     ShareGPT rows with prompt a2_v3, target = assembled unit Markdown.

  uv run python data/scripts/build_real_label_pool.py
  uv run python data/scripts/build_real_label_pool.py --skip-ocr
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.marker import (  # noqa: E402
    CURRENT_MARKER_TAG,
    draw_crosshair,
    sample_marker_area_frac,
    spec_for_image,
)
from point_ocr.prompts import pixel_to_norm  # noqa: E402
from point_ocr.sample_points import BBox, sample_diamond5_points  # noqa: E402

POOL_ID = "real_labeled"
DEFAULT_POOLS_ROOT = ROOT / "data" / "pools_real"
DEFAULT_LABEL_OUT = ROOT / "data" / "label_out"
DEFAULT_OCR_OUT = ROOT / "data" / "label_out_ocr"


def _run_ocr(*, label_dir: Path, ocr_dir: Path, model: str, force: bool) -> None:
    need = force or not ocr_dir.is_dir() or not any(ocr_dir.glob("*.json"))
    if not need:
        # Rebuild if label_out is newer than ocr outputs
        label_mtime = max((p.stat().st_mtime for p in label_dir.glob("*.json")), default=0)
        ocr_mtime = max((p.stat().st_mtime for p in ocr_dir.glob("*.json")), default=0)
        need = label_mtime > ocr_mtime
    if not need:
        print(f"[ocr] reuse {ocr_dir}", flush=True)
        return
    cmd = [
        sys.executable,
        str(ROOT / "data/scripts/ocr_label_boxes.py"),
        "--label-dir",
        str(label_dir),
        "--out",
        str(ocr_dir),
        "--model",
        model,
    ]
    print("[ocr] running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def _bbox_to_box(bb: list[Any]) -> BBox | None:
    if len(bb) != 4:
        return None
    x1, y1, x2, y2 = map(float, bb)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return BBox(x1, y1, x2, y2)


def _units_from_ann(ann: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer ``units`` from OCR pipeline; else group boxes by unit_id."""
    units = ann.get("units")
    if isinstance(units, list) and units:
        return [u for u in units if isinstance(u, dict)]
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, box in enumerate(ann.get("boxes") or []):
        uid = str(box.get("unit_id") or f"u{i + 1}")
        by[uid].append(box)
    out: list[dict[str, Any]] = []
    for uid, members in by.items():
        frags = [m.get("bbox") for m in members if m.get("bbox")]
        text = ""
        for m in members:
            t = str(m.get("unit_text") or m.get("text") or "").strip()
            if t:
                text = t
                break
        out.append(
            {
                "unit_id": uid,
                "kind": "multi_frag" if len(frags) > 1 else "single",
                "frag_bboxes": frags,
                "text": text,
                "manual": any(m.get("manual") for m in members),
            }
        )
    return out


def build_pool(
    *,
    ocr_dir: Path,
    pools_root: Path,
    seed: int,
) -> dict[str, Any]:
    pool_dir = pools_root / POOL_ID
    marked_dir = pool_dir / "marked"
    render_dir = pool_dir / "renders"
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    marked_dir.mkdir(parents=True)
    render_dir.mkdir(parents=True)

    rng = random.Random(seed)
    samples: list[PointSample] = []
    n_ann = 0
    n_units_ok = 0
    n_units_skip = 0
    n_multi = 0

    for fp in sorted(ocr_dir.glob("*.json")):
        if fp.name == "ocr_meta.json":
            continue
        ann = json.loads(fp.read_text(encoding="utf-8"))
        src = Path(str(ann.get("source_path") or ""))
        if not src.is_file():
            print(f"  skip missing image {fp.name}: {src}", flush=True)
            n_units_skip += 1
            continue
        img = Image.open(src).convert("RGB")
        w, h = img.size
        page_stem = fp.stem
        # Stable render copy (unmarked) for coord rewrite later
        render_path = render_dir / f"{page_stem}.png"
        img.save(render_path, format="PNG")
        n_ann += 1

        # Page occupancy from labeled ink (informational; None also OK for compose)
        ink = 0.0
        for u in _units_from_ann(ann):
            for bb in u.get("frag_bboxes") or []:
                box = _bbox_to_box(bb)
                if box:
                    ink += box.area()
        occ = min(1.0, ink / max(w * h, 1))

        source_rel = str(ann.get("source_rel") or src.name)
        source_folder = source_rel.split("/")[0] if "/" in source_rel else "real"

        for u in _units_from_ann(ann):
            text = str(u.get("text") or "").strip()
            if not text:
                n_units_skip += 1
                continue
            frags_raw = u.get("frag_bboxes") or []
            frags = [b for b in (_bbox_to_box(bb) for bb in frags_raw) if b is not None]
            if not frags:
                n_units_skip += 1
                continue
            unit_id = str(u.get("unit_id") or "u")
            kind = str(u.get("kind") or ("multi_frag" if len(frags) > 1 else "single"))
            if kind == "multi_frag" or len(frags) > 1:
                n_multi += 1
            n_units_ok += 1
            union = BBox(
                min(b.x0 for b in frags),
                min(b.y0 for b in frags),
                max(b.x1 for b in frags),
                max(b.y1 for b in frags),
            )
            # diamond5 on each fragment (click any frag → full unit text)
            for fi, frag in enumerate(frags):
                pts = sample_diamond5_points(
                    frag,
                    rng=rng,
                    block_id=f"{unit_id}_f{fi}",
                    image_w=w,
                    image_h=h,
                    fragment_index=fi,
                )
                for j, pt in enumerate(pts):
                    frac = sample_marker_area_frac(rng)
                    marked = draw_crosshair(img, pt.x, pt.y, spec_for_image(w, h, area_frac=frac))
                    name = f"{page_stem}__{unit_id}__f{fi}__p{j}.jpg"
                    path = marked_dir / name
                    marked.save(path, format="JPEG", quality=92)
                    nx, ny = pixel_to_norm(pt.x, pt.y, w, h)
                    sid = f"{POOL_ID}__{page_stem}:{unit_id}:f{fi}:p{j}"
                    meta = {
                        "pool_id": POOL_ID,
                        "prompt_key": "a2_v3",
                        "marker": CURRENT_MARKER_TAG,
                        "page_id": f"{POOL_ID}__{page_stem}",
                        "block_id": unit_id,
                        "point": [pt.x, pt.y],
                        "point_norm": [nx, ny],
                        "image_w": w,
                        "image_h": h,
                        "region": pt.region,
                        "bbox": list(frag.as_tuple()),
                        "bboxes": [list(b.as_tuple()) for b in frags],
                        "bbox_union": list(union.as_tuple()),
                        "n_fragments": len(frags),
                        "fragment_index": fi,
                        "is_negative": False,
                        "marker_area_frac": float(frac),
                        "a2_kind": kind,
                        "chrome_family": "none",
                        "template_stem": source_folder,
                        "source_path": str(src.resolve()),
                        "source_rel": source_rel,
                        "label_file": fp.name,
                        "text_occupancy": round(occ, 4),
                        "manual_unit": bool(u.get("manual")),
                    }
                    samples.append(
                        PointSample(
                            sample_id=sid,
                            image_path=str(path.resolve()),
                            task="POINT",
                            target=text,
                            meta=meta,
                        )
                    )

    jsonl = pool_dir / "point_sharegpt.jsonl"
    n_rows = write_jsonl(jsonl, samples)
    meta = {
        "pool_id": POOL_ID,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_annotation_files": n_ann,
        "n_units_ok": n_units_ok,
        "n_units_skip": n_units_skip,
        "n_multi_frag_units": n_multi,
        "n_samples": n_rows,
        "seed": seed,
        "prompt_key": "a2_v3",
        "coverage": "diamond5",
        "marker": CURRENT_MARKER_TAG,
        "ocr_dir": str(ocr_dir),
        "jsonl": str(jsonl),
    }
    (pool_dir / "pool_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_OUT)
    ap.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_OUT)
    ap.add_argument("--pools-root", type=Path, default=DEFAULT_POOLS_ROOT)
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--force-ocr", action="store_true")
    args = ap.parse_args()

    if not args.skip_ocr:
        _run_ocr(
            label_dir=args.label_dir,
            ocr_dir=args.ocr_dir,
            model=args.model,
            force=args.force_ocr,
        )
    elif not any(args.ocr_dir.glob("*.json")):
        raise SystemExit(f"no OCR json under {args.ocr_dir}; drop --skip-ocr")

    build_pool(ocr_dir=args.ocr_dir, pools_root=args.pools_root, seed=args.seed)


if __name__ == "__main__":
    main()

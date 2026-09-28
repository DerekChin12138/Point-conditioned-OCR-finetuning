#!/usr/bin/env python3
"""Build Stage-A2 POINT dataset (curriculum slices, A2 scatter-star marker).

Writes under a parallel tree so A1 ``data/processed/synth`` + ``data/splits``
are never touched::

  data/processed/a2/{filled_html,renders,marked,point_sharegpt.jsonl,build_meta.json}
  data/splits_a2/{train,val,test}.jsonl   # via companion split command

Example:
  export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
  uv run python data/scripts/build_a2_dataset.py --target 18000 --workers 6 --wipe
  uv run python data/scripts/split_train_val_test.py \\
    --input data/processed/a2/point_sharegpt.jsonl \\
    --out-dir data/splits_a2 \\
    --stratify-key a2_slice --interleave-template
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from PIL import Image  # noqa: E402

from apply_content_pack import fill_from_pool  # noqa: E402
from point_ocr.a2_labels import (  # noqa: E402
    build_semantic_group_map,
    is_block_visibly_clear,
    is_focused_window_block,
    parse_semantic_groups,
    rewrite_block_for_a2,
    select_desktop_positive_blocks,
)
from point_ocr.a2_mix import (  # noqa: E402
    A2_SLICE_FRAC,
    A2_SLICE_TEMPLATES,
    A2_TARGET_DEFAULT,
    interleave_a2_samples,
    meta_template_stem,
    quota_counts,
    summarize_mix,
    template_stem_from_page_id,
)
from point_ocr.build_point import BlockAnno, build_point_samples_for_page, load_blocks_json  # noqa: E402
from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.filter_qa import filter_block_label  # noqa: E402
from point_ocr.marker import MARKER_SPEC_A2, draw_scatter_star  # noqa: E402
from point_ocr.noise import apply_screen_noise  # noqa: E402
from point_ocr.prompts import pixel_to_norm  # noqa: E402
from point_ocr.sample_points import BBox, sample_points_in_block, sample_points_in_fragments  # noqa: E402
from point_ocr.synth.render import HtmlRenderSession  # noqa: E402

_WORKER: dict[str, Any] = {}

# Slices that fill HTML from the content pool (vs static UI shells).
_POOL_FILL_SLICES = frozenset({"replay", "adjacency", "multi_frag", "corner_extreme"})
_STATIC_SLICES = frozenset({"semantic_group", "special", "desktop", "near_edge"})


def _worker_init(pool_path: str) -> None:
    data = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    _WORKER["pool"] = data.get("pool") or data
    _WORKER["session"] = HtmlRenderSession(wait_until="load").__enter__()


def _worker_close() -> None:
    session = _WORKER.pop("session", None)
    if session is not None:
        session.__exit__(None, None, None)


def _dict_to_sample(d: dict[str, Any]) -> PointSample:
    return PointSample(
        sample_id=d["sample_id"],
        image_path=d["image_path"],
        task=d["task"],
        target=d["target"],
        meta=d.get("meta") or {},
    )


def _sample_to_dict(s: PointSample) -> dict[str, Any]:
    return asdict(s)


def _tag_meta(s: PointSample, *, slice_name: str, template_stem: str) -> PointSample:
    s.meta["a2_slice"] = slice_name
    s.meta["marker"] = "a2"
    s.meta["template_stem"] = template_stem
    return s


def _large_blocks(
    blocks: list[BlockAnno],
    page_area: float,
    *,
    min_area: float = 0.015,
    min_chars: int = 40,
) -> list[BlockAnno]:
    kept: list[BlockAnno] = []
    for b in blocks:
        fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
        if not fr.keep:
            continue
        if b.ink_area() / page_area < min_area:
            continue
        if len((b.markdown or "").strip()) < min_chars:
            continue
        kept.append(b)
    kept.sort(key=lambda b: b.ink_area(), reverse=True)
    return kept


def _emit_one(
    img: Image.Image,
    block: BlockAnno,
    *,
    page_id: str,
    out_dir: Path,
    coverage: str,
    seed: int,
    slice_name: str,
    idx: int,
    template_stem: str,
    allow_empty: bool = False,
) -> PointSample | None:
    md = (block.markdown or "").strip()
    if not md and not allow_empty:
        return None
    w, h = img.size
    frags = block.ink_rects()
    rng = random.Random(seed + idx)
    if len(frags) > 1:
        pts = sample_points_in_fragments(
            frags,
            n=1,
            coverage=coverage,
            rng=rng,
            block_id=block.block_id,
            image_w=w,
            image_h=h,
        )
    else:
        box = frags[0] if frags else block.bbox
        pts = sample_points_in_block(
            box,
            n=1,
            coverage=coverage,
            rng=rng,
            block_id=block.block_id,
            image_w=w,
            image_h=h,
            fragment_index=0,
        )
    if not pts:
        return None
    pt = pts[0]
    marked = draw_scatter_star(img, pt.x, pt.y, MARKER_SPEC_A2)
    name = f"{page_id}__{slice_name}__{block.block_id}__p{idx}.jpg"
    path = out_dir / name
    marked.save(path, format="JPEG", quality=92)
    nx, ny = pixel_to_norm(pt.x, pt.y, w, h)
    frag = pt.fragment_bbox or block.bbox
    meta = {
        "page_id": page_id,
        "block_id": block.block_id,
        "point": [pt.x, pt.y],
        "point_norm": [nx, ny],
        "image_w": w,
        "image_h": h,
        "region": pt.region,
        "bbox": list(frag.as_tuple()),
        "bboxes": [list(r.as_tuple()) for r in frags],
        "bbox_union": list(block.bbox.as_tuple()),
        "n_fragments": len(frags),
        "fragment_index": pt.fragment_index,
        "is_negative": False,
        "a2_slice": slice_name,
        "marker": "a2",
        "template_stem": template_stem,
    }
    if block.extra:
        for k in ("a2_kind", "group_id", "group_role", "group_kind", "visible_frac", "window_id", "window_focused"):
            if k in block.extra:
                meta[k] = block.extra[k]
    return PointSample(
        sample_id=f"{page_id}:{slice_name}:{block.block_id}:p{idx}",
        image_path=str(path.resolve()),
        task="POINT",
        target=block.markdown,
        meta=meta,
    )


def _neg_near_edge(
    img: Image.Image,
    avoid: list[BBox],
    *,
    page_id: str,
    out_dir: Path,
    seed: int,
    n: int,
    template_stem: str,
    band: float = 36.0,
) -> list[PointSample]:
    rng = random.Random(seed)
    w, h = img.size
    out: list[PointSample] = []
    tries = 0
    while len(out) < n and tries < 800:
        tries += 1
        side = rng.choice(["n", "s", "w", "e"])
        if side == "n":
            x, y = rng.uniform(band, w - band), rng.uniform(4, band)
        elif side == "s":
            x, y = rng.uniform(band, w - band), rng.uniform(h - band, h - 4)
        elif side == "w":
            x, y = rng.uniform(4, band), rng.uniform(band, h - band)
        else:
            x, y = rng.uniform(w - band, w - 4), rng.uniform(band, h - band)
        if any(b.contains(x, y) for b in avoid):
            continue
        marked = draw_scatter_star(img, x, y, MARKER_SPEC_A2)
        path = out_dir / f"{page_id}__near_edge__n{len(out)}.jpg"
        marked.save(path, format="JPEG", quality=92)
        nx, ny = pixel_to_norm(x, y, w, h)
        out.append(
            PointSample(
                sample_id=f"{page_id}:near_edge:n{len(out)}",
                image_path=str(path.resolve()),
                task="POINT",
                target="",
                meta={
                    "page_id": page_id,
                    "block_id": None,
                    "point": [x, y],
                    "point_norm": [nx, ny],
                    "image_w": w,
                    "image_h": h,
                    "region": f"near_edge_{side}",
                    "is_negative": True,
                    "a2_slice": "near_edge",
                    "marker": "a2",
                    "template_stem": template_stem,
                },
            )
        )
    return out


def _prepare_html(job: dict[str, Any], pool: dict) -> tuple[Path, str]:
    """Write filled/static HTML; return (path, template_stem)."""
    tmpl_path = Path(job["template"])
    filled_dir = Path(job["filled_dir"])
    page_id = job["page_id"]
    seed = int(job["seed"])
    i = int(job["index"])
    slice_name = job["slice"]
    stem = tmpl_path.stem
    filled_path = filled_dir / f"{page_id}.html"
    if slice_name in _POOL_FILL_SLICES or job.get("force_pool_fill"):
        html_src = tmpl_path.read_text(encoding="utf-8")
        filled, _layout = fill_from_pool(
            html_src,
            pool,
            random.Random(seed + i * 17),
            extra_paragraphs=int(job.get("extras", 5)),
            layout_set=str(job.get("layout_set", "dense")),
        )
        filled_path.write_text(filled, encoding="utf-8")
    else:
        filled_path.write_text(tmpl_path.read_text(encoding="utf-8"), encoding="utf-8")
    return filled_path, stem


def _rewrite_blocks(blocks: list[BlockAnno], page_html: str) -> list[BlockAnno]:
    gmap = build_semantic_group_map(page_html)
    return [rewrite_block_for_a2(b, page_html, gmap) for b in blocks]


def _process_job(job: dict[str, Any]) -> dict[str, Any]:
    """One page → list of sample dicts for a single curriculum slice."""
    pool = _WORKER["pool"]
    session: HtmlRenderSession = _WORKER["session"]
    slice_name = job["slice"]
    page_id = job["page_id"]
    seed = int(job["seed"])
    i = int(job["index"])
    use_noise = bool(job.get("use_noise", True))
    marked_dir = Path(job["marked_dir"])
    render_dir = Path(job["render_dir"])

    try:
        filled_path, template_stem = _prepare_html(job, pool)
        meta = session.render(filled_path, render_dir, page_id=page_id)
        img = Image.open(meta["image_path"]).convert("RGB")
        if use_noise and slice_name in _POOL_FILL_SLICES:
            img = apply_screen_noise(img, rng=random.Random(seed + i * 31))
        elif use_noise and slice_name in {"desktop", "semantic_group", "special"} and job.get("noise_static"):
            img = apply_screen_noise(img, rng=random.Random(seed + i * 31))

        page_html = filled_path.read_text(encoding="utf-8")
        blocks = _rewrite_blocks(load_blocks_json(Path(meta["blocks_path"])), page_html)
        all_boxes = [r for b in blocks for r in b.ink_rects()]
        w, h = img.size
        page_area = float(w * h)
        samples: list[PointSample] = []

        if slice_name == "replay":
            kept = _large_blocks(blocks, page_area)[:6]
            batch = build_point_samples_for_page(
                img,
                kept,
                page_id=page_id,
                out_image_dir=marked_dir,
                r_min=1,
                r_max=2,
                n_negatives=1,
                seed=seed + i,
                avoid_boxes=all_boxes,
                coverage="center_band",
                marker="a2",
                meta_extra={"a2_slice": "replay"},
            )
            for s in batch:
                _tag_meta(s, slice_name="replay", template_stem=template_stem)
                samples.append(s)

        elif slice_name == "adjacency":
            kept = _large_blocks(blocks, page_area, min_area=0.01, min_chars=30)[:6]
            batch = build_point_samples_for_page(
                img,
                kept,
                page_id=page_id,
                out_image_dir=marked_dir,
                r_min=1,
                r_max=2,
                n_negatives=1,
                seed=seed + i,
                avoid_boxes=all_boxes,
                coverage="edge_band",
                marker="a2",
                meta_extra={"a2_slice": "adjacency"},
            )
            for s in batch:
                _tag_meta(s, slice_name="adjacency", template_stem=template_stem)
                samples.append(s)

        elif slice_name == "multi_frag":
            multi = [
                b
                for b in blocks
                if len(b.ink_rects()) > 1 and len((b.markdown or "").strip()) >= 30
            ]
            multi.sort(key=lambda b: b.ink_area(), reverse=True)
            kept = multi[:6] or _large_blocks(blocks, page_area)[:3]
            # Prefer multi-point via build_point when we have multi-rect ink.
            if multi:
                batch = build_point_samples_for_page(
                    img,
                    kept,
                    page_id=page_id,
                    out_image_dir=marked_dir,
                    r_min=2,
                    r_max=3,
                    n_negatives=0,
                    seed=seed + i,
                    avoid_boxes=all_boxes,
                    coverage="center_band",
                    marker="a2",
                    meta_extra={"a2_slice": "multi_frag"},
                )
                for s in batch:
                    if s.meta.get("is_negative"):
                        continue
                    _tag_meta(s, slice_name="multi_frag", template_stem=template_stem)
                    samples.append(s)
            else:
                for j, b in enumerate(kept):
                    for k in range(2):
                        s = _emit_one(
                            img,
                            b,
                            page_id=page_id,
                            out_dir=marked_dir,
                            coverage="center_band",
                            seed=seed + i * 10 + k,
                            slice_name="multi_frag",
                            idx=j * 2 + k,
                            template_stem=template_stem,
                        )
                        if s:
                            samples.append(s)

        elif slice_name == "corner_extreme":
            kept = _large_blocks(blocks, page_area)[:5]
            batch = build_point_samples_for_page(
                img,
                kept,
                page_id=page_id,
                out_image_dir=marked_dir,
                r_min=1,
                r_max=1,
                n_negatives=0,
                seed=seed + i,
                avoid_boxes=all_boxes,
                coverage="corner_extreme",
                marker="a2",
                meta_extra={"a2_slice": "corner_extreme"},
            )
            for s in batch:
                if s.meta.get("is_negative"):
                    continue
                _tag_meta(s, slice_name="corner_extreme", template_stem=template_stem)
                samples.append(s)

        elif slice_name == "semantic_group":
            by_id = {b.block_id: b for b in blocks}
            groups = parse_semantic_groups(page_html)
            rng = random.Random(seed + i)
            n_take = min(12, max(1, len(groups)))
            order = list(range(len(groups)))
            rng.shuffle(order)
            for gi, g_idx in enumerate(order[:n_take]):
                g = groups[g_idx]
                # ~40% seed / ~60% body
                pick_body = rng.random() >= 0.40
                cand_ids = g.body_ids if pick_body and g.body_ids else g.seed_ids
                if not cand_ids:
                    cand_ids = g.member_ids
                if not cand_ids:
                    continue
                bid = cand_ids[gi % len(cand_ids)]
                src = by_id.get(bid)
                if src is None:
                    continue
                labeled = BlockAnno(
                    block_id=src.block_id,
                    bbox=src.bbox,
                    markdown=g.markdown,
                    rects=src.rects,
                    extra={
                        **(src.extra or {}),
                        "a2_kind": "semantic_group",
                        "group_id": g.group_id,
                        "group_kind": g.kind,
                        "group_role": "body" if pick_body else "seed",
                    },
                )
                s = _emit_one(
                    img,
                    labeled,
                    page_id=page_id,
                    out_dir=marked_dir,
                    coverage="center_band",
                    seed=seed + 1000 + gi,
                    slice_name="semantic_group",
                    idx=gi,
                    template_stem=template_stem,
                )
                if s:
                    samples.append(s)

        elif slice_name == "special":
            rng = random.Random(seed + i)
            # Prefer kind diversity: deny / table / prose
            by_kind: dict[str, list[BlockAnno]] = defaultdict(list)
            skip_ids = {
                "sp_title",
                "sp_meta",
                "sp_h_table",
                "sp_h_formula",
                "sp_h_code",
                "sp_h_img",
            }
            for b in blocks:
                kind = str((b.extra or {}).get("a2_kind") or "text")
                if b.block_id in skip_ids:
                    continue
                if kind in {"formula", "code", "image"}:
                    by_kind["deny"].append(
                        BlockAnno(
                            block_id=b.block_id,
                            bbox=b.bbox,
                            markdown="",
                            rects=b.rects,
                            extra={**(b.extra or {}), "a2_kind": kind},
                        )
                    )
                elif kind == "table" or (b.markdown or "").lstrip().lower().startswith("<table"):
                    by_kind["table"].append(b)
                elif len((b.markdown or "").strip()) >= 20:
                    by_kind["prose"].append(b)
            picks: list[BlockAnno] = []
            for bucket in ("deny", "table", "prose"):
                bucket_list = by_kind.get(bucket) or []
                rng.shuffle(bucket_list)
                picks.extend(bucket_list[:4])
            if not picks:
                picks = [b for b in blocks if len((b.markdown or "").strip()) >= 10][:6]
            for j, b in enumerate(picks[:12]):
                allow_empty = str((b.extra or {}).get("a2_kind") or "") in {
                    "formula",
                    "code",
                    "image",
                }
                s = _emit_one(
                    img,
                    b,
                    page_id=page_id,
                    out_dir=marked_dir,
                    coverage="center_band",
                    seed=seed + 2000 + j,
                    slice_name="special",
                    idx=j,
                    template_stem=template_stem,
                    allow_empty=allow_empty,
                )
                if s:
                    samples.append(s)

        elif slice_name == "desktop":
            candidates: list[BlockAnno] = []
            for b in blocks:
                kind = str((b.extra or {}).get("a2_kind") or "")
                if kind in {"formula", "code", "image"}:
                    continue
                if len((b.markdown or "").strip()) < 2:
                    continue
                candidates.append(b)
            group_cands = [
                b
                for b in candidates
                if (b.extra or {}).get("a2_kind") == "semantic_group"
                and is_block_visibly_clear(b.extra, min_visible_frac=0.9)
            ]
            focus_groups = [b for b in group_cands if is_focused_window_block(b.extra)]
            pool_sel = focus_groups or group_cands or candidates
            kept = select_desktop_positive_blocks(
                pool_sel,
                max_n=5,
                min_visible_frac=0.9,
                focused_frac=0.85,
            )
            kept = [
                b
                for b in kept
                if (b.extra or {}).get("visible_frac") is None
                or float(b.extra.get("visible_frac") or 0) >= 0.9
            ][:5]
            batch = build_point_samples_for_page(
                img,
                kept,
                page_id=page_id,
                out_image_dir=marked_dir,
                r_min=1,
                r_max=1,
                n_negatives=2,
                seed=seed + i,
                avoid_boxes=all_boxes,
                coverage="center_band",
                marker="a2",
                meta_extra={"a2_slice": "desktop"},
            )
            for s in batch:
                _tag_meta(s, slice_name="desktop", template_stem=template_stem)
                if not s.meta.get("is_negative"):
                    bid = s.meta.get("block_id")
                    src = next((b for b in kept if b.block_id == bid), None)
                    if src and src.extra:
                        s.meta["visible_frac"] = src.extra.get("visible_frac")
                        s.meta["window_id"] = src.extra.get("window_id")
                        s.meta["window_focused"] = src.extra.get("window_focused")
                        for k in ("a2_kind", "group_id", "group_role", "group_kind"):
                            if k in (src.extra or {}):
                                s.meta[k] = src.extra[k]
                samples.append(s)

        elif slice_name == "near_edge":
            n_neg = int(job.get("n_near_edge", 6))
            samples.extend(
                _neg_near_edge(
                    img,
                    all_boxes,
                    page_id=page_id,
                    out_dir=marked_dir,
                    seed=seed + i,
                    n=n_neg,
                    template_stem=template_stem,
                )
            )
        else:
            return {"page_id": page_id, "slice": slice_name, "skipped": True, "samples": [], "error": f"unknown slice {slice_name}"}

        return {
            "page_id": page_id,
            "slice": slice_name,
            "skipped": len(samples) == 0,
            "samples": [_sample_to_dict(s) for s in samples],
            "n": len(samples),
        }
    except Exception as e:
        return {
            "page_id": page_id,
            "slice": slice_name,
            "skipped": True,
            "samples": [],
            "error": f"{type(e).__name__}: {e}",
        }


def _resolve_templates(templates_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for slice_name, names in A2_SLICE_TEMPLATES.items():
        paths = [templates_dir / n for n in names if (templates_dir / n).is_file()]
        if not paths:
            raise SystemExit(f"No templates for slice {slice_name} under {templates_dir}")
        out[slice_name] = paths
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--templates-dir", type=Path, default=ROOT / "data/synth/templates")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/processed/a2",
        help="A2 output root (parallel to data/processed/synth)",
    )
    ap.add_argument("--target", type=int, default=A2_TARGET_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", action="store_true", default=True)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--max-pages", type=int, default=12_000)
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(10, (os.cpu_count() or 4) // 2)),
        help="Parallel Chromium worker processes (each owns a browser), like A1",
    )
    ap.add_argument(
        "--wave-size",
        type=int,
        default=0,
        help="Max in-flight pages (default: workers * 8, same as A1)",
    )
    ap.add_argument("--wipe", action="store_true", help="Clear previous outputs under --out")
    ap.add_argument(
        "--split-out",
        type=Path,
        default=ROOT / "data/splits_a2",
        help="If set with --also-split, write train/val/test here (not data/splits)",
    )
    ap.add_argument(
        "--also-split",
        action="store_true",
        help="After build, run stratified+interleaved split into --split-out",
    )
    args = ap.parse_args()
    use_noise = args.noise and not args.no_noise
    n_workers = max(1, args.workers)
    # Keep a deep in-flight queue so Chromium workers never idle (A1 uses workers*8).
    max_inflight = args.wave_size or (n_workers * 8)

    if not args.pool.is_file():
        raise SystemExit(f"Missing pool {args.pool}")

    quotas = quota_counts(args.target)
    templates = _resolve_templates(args.templates_dir)
    filled_dir = args.out / "filled_html"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"

    if args.wipe and args.out.exists():
        # Never touch A1 trees
        for forbidden in (
            ROOT / "data/processed/synth",
            ROOT / "data/splits",
        ):
            if args.out.resolve() == forbidden.resolve():
                raise SystemExit(f"Refusing to wipe A1 path: {args.out}")
        shutil.rmtree(args.out)
        print(f"[wipe] removed {args.out}", flush=True)

    for d in (filled_dir, render_dir, marked_dir):
        d.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[PointSample]] = {k: [] for k in quotas}
    tmpl_rr: dict[str, int] = {k: 0 for k in quotas}
    in_flight: dict[str, int] = {k: 0 for k in quotas}
    pages_done = 0
    pages_failed = 0
    next_i = 0
    t0 = time.time()

    print(
        f"[start] A2 target={args.target} quotas={quotas} workers={n_workers} "
        f"inflight={max_inflight} out={args.out} (A1 untouched)",
        flush=True,
    )

    def _need(slice_name: str) -> int:
        # Pages still running — assume ~4 samples each so we don't over-queue.
        reserved = in_flight[slice_name] * 4
        return max(0, quotas[slice_name] - len(buckets[slice_name]) - reserved)

    def _enough() -> bool:
        return all(len(buckets[k]) >= quotas[k] for k in quotas)

    def _pick_slice() -> str | None:
        # Round-robin among underfilled slices (by remaining count), not absolute
        # quota size — otherwise replay (22%) monopolizes the pipeline.
        needy = [k for k in quotas if _need(k) > 0]
        if not needy:
            return None
        # Weight by remaining samples so larger slices get more pages, but every
        # slice still appears each cycle.
        weights = []
        for k in needy:
            rem = max(1, quotas[k] - len(buckets[k]))
            weights.append(rem)
        return random.Random(args.seed + next_i * 13).choices(needy, weights=weights, k=1)[0]

    def _make_job(slice_name: str, index: int) -> dict[str, Any]:
        tmpls = templates[slice_name]
        tmpl = tmpls[tmpl_rr[slice_name] % len(tmpls)]
        tmpl_rr[slice_name] += 1
        page_id = f"{slice_name}__{tmpl.stem}__{index}"
        extras = 6 if slice_name in {"replay", "adjacency", "multi_frag"} else 4
        return {
            "slice": slice_name,
            "template": str(tmpl),
            "page_id": page_id,
            "index": index,
            "seed": args.seed,
            "filled_dir": str(filled_dir),
            "render_dir": str(render_dir),
            "marked_dir": str(marked_dir),
            "use_noise": use_noise,
            "noise_static": slice_name in {"desktop", "semantic_group"} and (index % 3 == 0),
            "extras": extras,
            "layout_set": "dense",
            "n_near_edge": 12 if slice_name == "near_edge" else 0,
        }

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(args.pool.resolve()),),
    ) as ex:
        pending: dict[Any, str] = {}

        def _fill_pipeline() -> None:
            """Keep up to max_inflight pages queued — refill as soon as a worker frees."""
            nonlocal next_i
            while (
                len(pending) < max_inflight
                and not _enough()
                and next_i < args.max_pages
            ):
                slice_name = _pick_slice()
                if slice_name is None:
                    break
                job = _make_job(slice_name, next_i)
                fut = ex.submit(_process_job, job)
                pending[fut] = slice_name
                in_flight[slice_name] += 1
                next_i += 1

        _fill_pipeline()
        while pending and not _enough():
            # One-at-a-time completion + immediate refill (no wave barrier).
            done_map = as_completed(pending.keys())
            fut = next(done_map)
            slice_hint = pending.pop(fut, None)
            if slice_hint:
                in_flight[slice_hint] = max(0, in_flight[slice_hint] - 1)
            pages_done += 1
            try:
                result = fut.result()
            except Exception as e:
                pages_failed += 1
                print(f"[warn] future failed ({slice_hint}): {e}", flush=True)
                _fill_pipeline()
                continue
            if result.get("error"):
                pages_failed += 1
                if pages_failed <= 20 or pages_failed % 50 == 0:
                    print(f"[warn] {result.get('page_id')}: {result['error']}", flush=True)
            sl = str(result.get("slice") or slice_hint or "")
            if sl in buckets:
                room = max(0, quotas[sl] - len(buckets[sl]))
                for row in result.get("samples") or []:
                    if room <= 0:
                        break
                    s = _dict_to_sample(row)
                    s.meta["a2_slice"] = sl
                    if not s.meta.get("template_stem"):
                        s.meta["template_stem"] = template_stem_from_page_id(
                            str(s.meta.get("page_id") or "")
                        )
                    buckets[sl].append(s)
                    room -= 1

            if pages_done % 25 == 0 or _enough():
                elapsed = time.time() - t0
                filled = {k: len(v) for k, v in buckets.items()}
                total = sum(filled.values())
                rate = pages_done / elapsed if elapsed > 0 else 0
                samp_rate = total / elapsed if elapsed > 0 else 0
                print(
                    f"[ok] pages={pages_done} fail={pages_failed} "
                    f"{rate:.2f} pg/s {samp_rate:.1f} samp/s "
                    f"n={total}/{args.target} pending={len(pending)} "
                    f"filled={filled}",
                    flush=True,
                )

            if not _enough():
                _fill_pipeline()
            if _enough():
                break

        for fut in list(pending.keys()):
            fut.cancel()
            sl = pending.pop(fut, None)
            if sl:
                in_flight[sl] = max(0, in_flight[sl] - 1)

    short = {k: quotas[k] - len(buckets[k]) for k in quotas if len(buckets[k]) < quotas[k]}
    if short:
        print(f"[warn] under-quota slices: {short}", flush=True)

    all_samples: list[PointSample] = []
    for k in quotas:
        all_samples.extend(buckets[k][: quotas[k]])

    rng = random.Random(args.seed)
    all_samples = interleave_a2_samples(
        all_samples,
        rng=rng,
        slice_fn=lambda s: str(s.meta.get("a2_slice") or "?"),
        template_fn=lambda s: meta_template_stem(s.meta),
    )

    jsonl = args.out / "point_sharegpt.jsonl"
    n = write_jsonl(jsonl, all_samples)
    mix = summarize_mix([s.to_sharegpt() for s in all_samples])
    n_neg = sum(1 for s in all_samples if s.meta.get("is_negative"))
    elapsed = time.time() - t0
    build_meta = {
        "protocol": "a2_curriculum_scatter_star",
        "target": args.target,
        "n_written": n,
        "n_neg": n_neg,
        "neg_frac": round(n_neg / n, 4) if n else 0.0,
        "quotas": quotas,
        "slice_frac": dict(A2_SLICE_FRAC),
        "filled": {k: len(buckets[k]) for k in quotas},
        "pages_done": pages_done,
        "pages_failed": pages_failed,
        "workers": n_workers,
        "seed": args.seed,
        "out": str(args.out),
        "elapsed_sec": round(elapsed, 1),
        "mix": mix,
        "note": "Parallel to A1 data/processed/synth; do not overwrite A1 splits",
    }
    (args.out / "build_meta.json").write_text(
        json.dumps(build_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(build_meta, indent=2), flush=True)
    print(f"Wrote {n} → {jsonl}", flush=True)

    if n < int(0.95 * args.target):
        raise SystemExit(
            f"A2 build too short: got {n} need ~{args.target}; raise --max-pages or check templates"
        )

    if args.also_split:
        import subprocess

        args.split_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "data/scripts/split_train_val_test.py"),
            "--input",
            str(jsonl),
            "--out-dir",
            str(args.split_out),
            "--seed",
            str(args.seed),
            "--stratify-key",
            "a2_slice",
            "--interleave-template",
        ]
        print("[split]", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    try:
        main()
    finally:
        _worker_close()

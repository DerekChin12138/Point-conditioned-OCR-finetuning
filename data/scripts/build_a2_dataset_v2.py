#!/usr/bin/env python3
"""Build Stage-A2_v2 POINT dataset (same V2 translucent-dot marker as A1_v2).

Does not touch legacy A1/A2 trees::

  data/processed/a2_v2/...
  data/splits_a2_v2/...

Example:
  export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
  uv run python data/scripts/build_a2_dataset_v2.py --target 16000 --workers 30 --wipe --also-split
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
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
    block_fully_in_frame,
    parse_semantic_groups,
    rewrite_block_for_a2,
)
from point_ocr.a2_mix import (  # noqa: E402
    interleave_a2_samples,
    meta_template_stem,
    template_stem_from_page_id,
)
from point_ocr.a2_v2_mix import (  # noqa: E402
    A2_V2_SLICE_FRAC,
    A2_V2_SLICE_TEMPLATES,
    A2_V2_STATIC_SLICES,
    A2_V2_TARGET_DEFAULT,
    a2_v2_quota_counts,
)
from point_ocr.build_point import BlockAnno, build_point_samples_for_page, load_blocks_json  # noqa: E402
from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.filter_qa import filter_block_label  # noqa: E402
from point_ocr.marker import MARKER_SPEC, draw_crosshair  # noqa: E402
from point_ocr.noise import apply_screen_noise  # noqa: E402
from point_ocr.prompts import pixel_to_norm  # noqa: E402
from point_ocr.sample_points import sample_points_in_block, sample_points_in_fragments  # noqa: E402
from point_ocr.synth.render import HtmlRenderSession  # noqa: E402
from point_ocr.synth.window_viewport import sample_window_capture  # noqa: E402

_WORKER: dict[str, Any] = {}


def _worker_init(pool_path: str) -> None:
    data = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    _WORKER["pool"] = data.get("pool") or data
    _WORKER["session"] = HtmlRenderSession(wait_until="load").__enter__()


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


def _large_blocks(
    blocks: list[BlockAnno],
    page_area: float,
    *,
    image_w: int,
    image_h: int,
    min_area: float = 0.015,
    min_chars: int = 40,
) -> list[BlockAnno]:
    kept: list[BlockAnno] = []
    for b in blocks:
        fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
        if not fr.keep:
            continue
        if not block_fully_in_frame(b, image_w, image_h):
            continue
        if b.ink_area() / page_area < min_area:
            continue
        if len((b.markdown or "").strip()) < min_chars:
            continue
        kept.append(b)
    kept.sort(key=lambda b: b.ink_area(), reverse=True)
    return kept


def _pick_spread(blocks: list[BlockAnno], n: int, rng: random.Random) -> list[BlockAnno]:
    if len(blocks) <= n:
        return list(blocks)
    by_y = sorted(blocks, key=lambda b: b.bbox.y0)
    skip = max(1, len(by_y) // 5)
    pool = by_y[skip:] or by_y
    large = sorted(pool, key=lambda b: b.ink_area(), reverse=True)
    chosen: list[BlockAnno] = []
    for b in large:
        if b not in chosen:
            chosen.append(b)
        if len(chosen) >= n:
            break
    return chosen


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
            frags, n=1, coverage=coverage, rng=rng, block_id=block.block_id, image_w=w, image_h=h
        )
    else:
        box = frags[0] if frags else block.bbox
        pts = sample_points_in_block(
            box, n=1, coverage=coverage, rng=rng, block_id=block.block_id, image_w=w, image_h=h, fragment_index=0
        )
    if not pts:
        return None
    pt = pts[0]
    marked = draw_crosshair(img, pt.x, pt.y, MARKER_SPEC)
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
        "marker": "x45",
        "template_stem": template_stem,
    }
    if block.extra:
        for k in ("a2_kind", "group_id", "group_role", "group_kind"):
            if k in block.extra:
                meta[k] = block.extra[k]
    return PointSample(
        sample_id=f"{page_id}:{slice_name}:{block.block_id}:p{idx}",
        image_path=str(path.resolve()),
        task="POINT",
        target=block.markdown,
        meta=meta,
    )


def _process_job(job: dict[str, Any]) -> dict[str, Any]:
    pool = _WORKER["pool"]
    session: HtmlRenderSession = _WORKER["session"]
    slice_name = job["slice"]
    page_id = job["page_id"]
    seed = int(job["seed"])
    i = int(job["index"])
    use_noise = bool(job.get("use_noise", True))
    marked_dir = Path(job["marked_dir"])
    render_dir = Path(job["render_dir"])
    filled_dir = Path(job["filled_dir"])
    tmpl_path = Path(job["template"])
    template_stem = tmpl_path.stem

    try:
        if slice_name in A2_V2_STATIC_SLICES:
            filled_html = tmpl_path.read_text(encoding="utf-8")
            layout_name = "static"
        else:
            filled_html, layout_name = fill_from_pool(
                tmpl_path.read_text(encoding="utf-8"),
                pool,
                random.Random(seed + i * 17),
                extra_paragraphs=int(job.get("extras", 6)),
                layout_set="dense",
            )
        filled_path = filled_dir / f"{page_id}.html"
        filled_path.write_text(filled_html, encoding="utf-8")

        cap = sample_window_capture(random.Random(seed + i * 41))
        meta = session.render(
            filled_path,
            render_dir,
            page_id=page_id,
            viewport=cap.viewport,
            device_scale_factor=cap.device_scale_factor,
            full_page=cap.full_page,
        )
        img = Image.open(meta["image_path"]).convert("RGB")
        if use_noise and slice_name not in A2_V2_STATIC_SLICES:
            img = apply_screen_noise(img, rng=random.Random(seed + i * 31))
        elif use_noise and slice_name == "semantic_group" and (i % 4 == 0):
            img = apply_screen_noise(img, rng=random.Random(seed + i * 31))

        page_html = filled_html
        raw_blocks = load_blocks_json(Path(meta["blocks_path"]))
        gmap = build_semantic_group_map(page_html) if slice_name == "semantic_group" else None
        blocks = (
            [rewrite_block_for_a2(b, page_html, gmap) for b in raw_blocks]
            if slice_name in {"semantic_group", "special_light"}
            else raw_blocks
        )
        all_boxes = [r for b in blocks for r in b.ink_rects()]
        w, h = img.size
        page_area = float(w * h)
        rng = random.Random(seed + i * 7)
        samples: list[PointSample] = []
        capture_meta = {
            **cap.to_meta(),
            "render_viewport": list(meta.get("viewport") or cap.viewport),
            "render_device_scale_factor": meta.get("device_scale_factor", cap.device_scale_factor),
        }

        if slice_name == "replay_long":
            kept = _pick_spread(_large_blocks(blocks, page_area, image_w=w, image_h=h), 5, rng)
            batch = build_point_samples_for_page(
                img, kept, page_id=page_id, out_image_dir=marked_dir,
                r_min=1, r_max=2, n_negatives=0, seed=seed + i,
                avoid_boxes=all_boxes, coverage="center_band", marker="x45",
                meta_extra={"a2_slice": "replay_long", "layout": layout_name, "prompt_key": "a2_v2", **capture_meta},
            )
            samples = [s for s in batch if not s.meta.get("is_negative")]

        elif slice_name == "adjacency":
            kept = _pick_spread(_large_blocks(blocks, page_area, image_w=w, image_h=h, min_area=0.01, min_chars=30), 5, rng)
            batch = build_point_samples_for_page(
                img, kept, page_id=page_id, out_image_dir=marked_dir,
                r_min=1, r_max=2, n_negatives=0, seed=seed + i,
                avoid_boxes=all_boxes, coverage="edge_band", marker="x45",
                meta_extra={"a2_slice": "adjacency", "layout": layout_name, "prompt_key": "a2_v2", **capture_meta},
            )
            samples = [s for s in batch if not s.meta.get("is_negative")]

        elif slice_name == "multi_frag":
            multi = [
                b for b in blocks
                if len(b.ink_rects()) > 1
                and len((b.markdown or "").strip()) >= 30
                and block_fully_in_frame(b, w, h)
            ]
            multi.sort(key=lambda b: b.ink_area(), reverse=True)
            kept = multi[:5] or _pick_spread(_large_blocks(blocks, page_area, image_w=w, image_h=h), 3, rng)
            batch = build_point_samples_for_page(
                img, kept, page_id=page_id, out_image_dir=marked_dir,
                r_min=1, r_max=2, n_negatives=0, seed=seed + i,
                avoid_boxes=all_boxes, coverage="center_band", marker="x45",
                meta_extra={"a2_slice": "multi_frag", "layout": layout_name, "prompt_key": "a2_v2", **capture_meta},
            )
            samples = [s for s in batch if not s.meta.get("is_negative")]

        elif slice_name == "corner_extreme":
            kept = _pick_spread(_large_blocks(blocks, page_area, image_w=w, image_h=h), 4, rng)
            batch = build_point_samples_for_page(
                img, kept, page_id=page_id, out_image_dir=marked_dir,
                r_min=1, r_max=1, n_negatives=0, seed=seed + i,
                avoid_boxes=all_boxes, coverage="corner_extreme", marker="x45",
                meta_extra={"a2_slice": "corner_extreme", "layout": layout_name, "prompt_key": "a2_v2", **capture_meta},
            )
            samples = [s for s in batch if not s.meta.get("is_negative")]

        elif slice_name == "empty_neg":
            batch = build_point_samples_for_page(
                img, [], page_id=page_id, out_image_dir=marked_dir,
                r_min=1, r_max=1, n_negatives=int(job.get("n_negatives", 10)),
                seed=seed + i, avoid_boxes=all_boxes, coverage="center_band", marker="x45",
                meta_extra={"a2_slice": "empty_neg", "layout": layout_name, "prompt_key": "a2_v2", **capture_meta},
            )
            samples = [s for s in batch if s.meta.get("is_negative")]

        elif slice_name == "semantic_group":
            by_id = {b.block_id: b for b in blocks}
            groups = parse_semantic_groups(page_html)
            # Positive: group members with expanded GT
            order = list(range(len(groups)))
            rng.shuffle(order)
            n_take = min(8, max(1, len(groups)))
            for gi, g_idx in enumerate(order[:n_take]):
                g = groups[g_idx]
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
                members = [by_id[mid] for mid in g.member_ids if mid in by_id]
                # Group GT spans all members — reject if any member is window-cropped.
                if not members or not all(block_fully_in_frame(mb, w, h) for mb in members):
                    continue
                if not block_fully_in_frame(src, w, h):
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
                    img, labeled, page_id=page_id, out_dir=marked_dir,
                    coverage="center_band", seed=seed + 1000 + gi,
                    slice_name="semantic_group", idx=gi, template_stem=template_stem,
                )
                if s:
                    samples.append(s)

            # Truncation negatives: outside-group neighbors → GT = own text only
            def _is_trunc(b: BlockAnno) -> bool:
                ex = b.extra or {}
                if ex.get("truncate_neighbor") not in (None, "", "0", 0, False):
                    return True
                if b.block_id in {"orphan_nav", "d_outside", "n_outside", "w_outside", "r_outside"}:
                    return True
                if str(b.block_id).endswith("_outside") or str(b.block_id).endswith("outside"):
                    return True
                return False

            trunc = [b for b in raw_blocks if _is_trunc(b) and block_fully_in_frame(b, w, h)]
            if not trunc and gmap is not None:
                trunc = [
                    b for b in raw_blocks
                    if b.block_id not in gmap
                    and len((b.markdown or "").strip()) >= 12
                    and block_fully_in_frame(b, w, h)
                ][:2]
            rng.shuffle(trunc)
            for ti, b in enumerate(trunc[:3]):
                # Own markdown only (no group expand)
                labeled = BlockAnno(
                    block_id=b.block_id,
                    bbox=b.bbox,
                    markdown=b.markdown,
                    rects=b.rects,
                    extra={**(b.extra or {}), "a2_kind": "truncate_neighbor", "group_role": "outside"},
                )
                s = _emit_one(
                    img, labeled, page_id=page_id, out_dir=marked_dir,
                    coverage="center_band", seed=seed + 2000 + ti,
                    slice_name="semantic_group", idx=80 + ti, template_stem=template_stem,
                )
                if s:
                    s.meta["truncate_neg"] = True
                    samples.append(s)

        elif slice_name == "special_light":
            by_kind: dict[str, list[BlockAnno]] = defaultdict(list)
            skip = {"sp_title", "sp_meta", "sp_h_table", "sp_h_formula", "sp_h_code", "sp_h_img"}
            for b in blocks:
                if not block_fully_in_frame(b, w, h):
                    continue
                kind = str((b.extra or {}).get("a2_kind") or "text")
                if b.block_id in skip:
                    continue
                if kind in {"formula", "code", "image"}:
                    by_kind["deny"].append(
                        BlockAnno(
                            block_id=b.block_id, bbox=b.bbox, markdown="",
                            rects=b.rects, extra={**(b.extra or {}), "a2_kind": kind},
                        )
                    )
                elif kind == "table" or (b.markdown or "").lstrip().lower().startswith("<table"):
                    by_kind["table"].append(b)
                elif len((b.markdown or "").strip()) >= 20:
                    by_kind["prose"].append(b)
            picks: list[BlockAnno] = []
            for bucket in ("deny", "table", "prose"):
                bl = by_kind.get(bucket) or []
                rng.shuffle(bl)
                picks.extend(bl[:3])
            for j, b in enumerate(picks[:10]):
                allow_empty = str((b.extra or {}).get("a2_kind") or "") in {"formula", "code", "image"}
                s = _emit_one(
                    img, b, page_id=page_id, out_dir=marked_dir,
                    coverage="center_band", seed=seed + 3000 + j,
                    slice_name="special_light", idx=j, template_stem=template_stem,
                    allow_empty=allow_empty,
                )
                if s:
                    samples.append(s)
        else:
            return {"page_id": page_id, "slice": slice_name, "skipped": True, "samples": [], "error": f"unknown {slice_name}"}

        for s in samples:
            s.meta["a2_slice"] = slice_name
            s.meta["marker"] = "x45"
            s.meta["prompt_key"] = "a2_v2"
            s.meta["template_stem"] = template_stem
            s.meta.update(capture_meta)

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


def _default_render_workers() -> int:
    cpus = os.cpu_count() or 4
    return max(1, cpus - 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--templates-dir", type=Path, default=ROOT / "data/synth/templates")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/a2_v2")
    ap.add_argument("--target", type=int, default=A2_V2_TARGET_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", action="store_true", default=True)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--max-pages", type=int, default=14_000)
    ap.add_argument(
        "--workers",
        type=int,
        default=_default_render_workers(),
        help="Parallel Playwright workers (default: cpu_count-2)",
    )
    ap.add_argument("--wave-size", type=int, default=0, help="Max in-flight pages (default workers*12)")
    ap.add_argument("--wipe", action="store_true")
    ap.add_argument("--split-out", type=Path, default=ROOT / "data/splits_a2_v2")
    ap.add_argument("--also-split", action="store_true")
    args = ap.parse_args()

    use_noise = args.noise and not args.no_noise
    n_workers = max(1, args.workers)
    max_inflight = args.wave_size or (n_workers * 12)
    quotas = a2_v2_quota_counts(args.target)

    templates: dict[str, list[Path]] = {}
    for sl, names in A2_V2_SLICE_TEMPLATES.items():
        paths = [args.templates_dir / n for n in names if (args.templates_dir / n).is_file()]
        if not paths:
            raise SystemExit(f"No templates for {sl}")
        templates[sl] = paths

    filled_dir = args.out / "filled_html"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"

    protected = {
        (ROOT / "data/processed/synth").resolve(),
        (ROOT / "data/processed/a1_v2").resolve(),
        (ROOT / "data/splits").resolve(),
        (ROOT / "data/splits_a1_v2").resolve(),
    }
    if args.wipe and args.out.exists():
        if args.out.resolve() in protected:
            raise SystemExit(f"Refusing to wipe protected path {args.out}")
        shutil.rmtree(args.out)
        print(f"[wipe] removed {args.out}", flush=True)

    for d in (filled_dir, render_dir, marked_dir):
        d.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[PointSample]] = {k: [] for k in quotas}
    tmpl_rr = {k: 0 for k in quotas}
    in_flight = {k: 0 for k in quotas}
    pages_done = pages_failed = next_i = 0
    t0 = time.time()

    print(
        f"[start] A2_v2 target={args.target} quotas={quotas} workers={n_workers} "
        f"inflight={max_inflight} out={args.out}",
        flush=True,
    )

    def _need(sl: str) -> int:
        return max(0, quotas[sl] - len(buckets[sl]) - in_flight[sl] * 4)

    def _enough() -> bool:
        return all(len(buckets[k]) >= quotas[k] for k in quotas)

    def _pick_slice() -> str | None:
        needy = [k for k in quotas if _need(k) > 0]
        if not needy:
            return None
        weights = [max(1, quotas[k] - len(buckets[k])) for k in needy]
        return random.Random(args.seed + next_i * 13).choices(needy, weights=weights, k=1)[0]

    def _make_job(sl: str, index: int) -> dict[str, Any]:
        tmpls = templates[sl]
        tmpl = tmpls[tmpl_rr[sl] % len(tmpls)]
        tmpl_rr[sl] += 1
        return {
            "slice": sl,
            "template": str(tmpl),
            "page_id": f"{sl}__{tmpl.stem}__{index}",
            "index": index,
            "seed": args.seed,
            "filled_dir": str(filled_dir),
            "render_dir": str(render_dir),
            "marked_dir": str(marked_dir),
            "use_noise": use_noise,
            "extras": 6,
            "n_negatives": 10 if sl == "empty_neg" else 0,
        }

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(args.pool.resolve()),),
    ) as ex:
        pending: dict[Any, str] = {}

        def _fill_pipeline() -> None:
            nonlocal next_i
            while len(pending) < max_inflight and not _enough() and next_i < args.max_pages:
                sl = _pick_slice()
                if sl is None:
                    break
                fut = ex.submit(_process_job, _make_job(sl, next_i))
                pending[fut] = sl
                in_flight[sl] += 1
                next_i += 1

        _fill_pipeline()
        while pending and not _enough():
            fut = next(as_completed(pending.keys()))
            sl_hint = pending.pop(fut, None)
            if sl_hint:
                in_flight[sl_hint] = max(0, in_flight[sl_hint] - 1)
            pages_done += 1
            try:
                result = fut.result()
            except Exception as e:
                pages_failed += 1
                print(f"[warn] future failed ({sl_hint}): {e}", flush=True)
                _fill_pipeline()
                continue
            if result.get("error"):
                pages_failed += 1
                if pages_failed <= 20 or pages_failed % 50 == 0:
                    print(f"[warn] {result.get('page_id')}: {result['error']}", flush=True)
            sl = str(result.get("slice") or sl_hint or "")
            if sl in buckets:
                room = max(0, quotas[sl] - len(buckets[sl]))
                for row in result.get("samples") or []:
                    if room <= 0:
                        break
                    s = _dict_to_sample(row)
                    s.meta["a2_slice"] = sl
                    s.meta.setdefault("marker", "v2")
                    if not s.meta.get("template_stem"):
                        s.meta["template_stem"] = template_stem_from_page_id(str(s.meta.get("page_id") or ""))
                    buckets[sl].append(s)
                    room -= 1

            if pages_done % 25 == 0 or _enough():
                elapsed = time.time() - t0
                filled = {k: len(v) for k, v in buckets.items()}
                total = sum(filled.values())
                print(
                    f"[ok] pages={pages_done} fail={pages_failed} "
                    f"{pages_done/elapsed:.2f} pg/s {total/elapsed:.1f} samp/s "
                    f"n={total}/{args.target} pending={len(pending)} filled={filled}",
                    flush=True,
                )
            if not _enough():
                _fill_pipeline()
            if _enough():
                break
        for fut in list(pending.keys()):
            fut.cancel()

    short = {k: quotas[k] - len(buckets[k]) for k in quotas if len(buckets[k]) < quotas[k]}
    if short:
        print(f"[warn] under-quota: {short}", flush=True)

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
    n_neg = sum(1 for s in all_samples if s.meta.get("is_negative"))
    n_trunc = sum(1 for s in all_samples if s.meta.get("truncate_neg"))
    elapsed = time.time() - t0
    build_meta = {
        "protocol": "a2_v2_window_capture_diverse_res",
        "marker_spec": {
            "tag": "x45",
            "rotation_deg": 45.0,
            "cross_half_length_px": 34,
            "cross_width_px": 3,
            "ring_radius_px": 28,
            "fixed_magenta": True,
        },
        "target": args.target,
        "n_written": n,
        "n_neg": n_neg,
        "n_truncate_neg": n_trunc,
        "neg_frac": round(n_neg / n, 4) if n else 0.0,
        "quotas": quotas,
        "slice_frac": dict(A2_V2_SLICE_FRAC),
        "filled": {k: len(buckets[k]) for k in quotas},
        "by_slice": dict(sorted(Counter(s.meta.get("a2_slice") for s in all_samples).items())),
        "by_template": dict(sorted(Counter(meta_template_stem(s.meta) for s in all_samples).items())),
        "pages_done": pages_done,
        "pages_failed": pages_failed,
        "workers": n_workers,
        "seed": args.seed,
        "out": str(args.out),
        "elapsed_sec": round(elapsed, 1),
    }
    (args.out / "build_meta.json").write_text(json.dumps(build_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(build_meta, indent=2), flush=True)
    print(f"Wrote {n} → {jsonl}", flush=True)

    if n < int(0.95 * args.target):
        raise SystemExit(f"A2_v2 too short: got {n} need ~{args.target}")

    if args.also_split:
        args.split_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "data/scripts/split_train_val_test.py"),
            "--input", str(jsonl),
            "--out-dir", str(args.split_out),
            "--seed", str(args.seed),
            "--stratify-key", "a2_slice",
            "--interleave-template",
        ]
        print("[split]", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

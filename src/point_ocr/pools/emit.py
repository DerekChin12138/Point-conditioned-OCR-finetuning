"""Emit POINT samples for one rendered page into a given pool."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image

from point_ocr.a2_labels import block_fully_in_frame, classify_a2_kind, parse_semantic_groups
from point_ocr.build_point import BlockAnno, build_point_samples_for_page
from point_ocr.dataset_format import PointSample
from point_ocr.marker import (
    CURRENT_MARKER_TAG,
    MARKER_AREA_FRAC,
    MARKER_AREA_FRAC_LEGACY,
    area_frac_for_scale,
    draw_crosshair,
    sample_marker_area_frac,
    spec_for_image,
)
from point_ocr.ocr_mt import wrap_point_target
from point_ocr.pools.select import is_special_block, large_blocks, multi_frag_blocks, pick_spread
from point_ocr.pools.units import clustered_member_ids, occupancy_boxes
from point_ocr.prompts import pixel_to_norm
from point_ocr.sample_points import (
    Q1_NEG_CLEARANCE_PX,
    facing_gap_px,
    gap_strip,
    sample_boundary_empty_points,
    sample_diamond5_points,
    sample_points_in_block,
    sample_points_in_fragments,
    union_bbox,
)


def _base_meta(
    *,
    pool_id: str,
    template_stem: str,
    prompt_key: str,
    layout_name: str | None,
    capture_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pool_id": pool_id,
        "template_stem": template_stem,
        "prompt_key": prompt_key,
        "marker": CURRENT_MARKER_TAG,
        "layout": layout_name,
        **capture_meta,
    }


def _stamp(samples: list[PointSample], extra: dict[str, Any]) -> list[PointSample]:
    skip = {"marker_area_scale", "marker_area_frac"}
    for s in samples:
        for k, v in extra.items():
            if k in skip and k in s.meta:
                continue
            if k not in s.meta:
                s.meta[k] = v
        s.meta.setdefault("pool_id", extra.get("pool_id"))
    return samples


def _choose_area_frac(extra: dict[str, Any], rng: random.Random) -> float:
    """Page-area fraction for one marked image.

    Current protocol samples uniformly from [MARKER_AREA_FRAC_MIN, MAX] unless the
    caller pinned an explicit ``marker_area_frac`` / ``marker_area_scale``.
    """
    raw = extra.get("marker_area_frac")
    if raw is not None:
        return float(raw)
    raw = extra.get("marker_area_scale")
    if raw is not None:
        return area_frac_for_scale(float(raw))
    return sample_marker_area_frac(rng)


def _frac_meta(extra: dict[str, Any], area_frac: float) -> dict[str, Any]:
    return {
        **extra,
        "marker_area_frac": float(area_frac),
        "marker_area_scale": float(area_frac) / MARKER_AREA_FRAC_LEGACY,
    }


def _spec_for(w: int, h: int, area_frac: float):
    return spec_for_image(w, h, area_frac=float(area_frac))


def _wrap_md(markdown: str, extra: dict[str, Any], *, is_negative: bool = False) -> str:
    zh = ""
    if not is_negative:
        zh = str(extra.get("translation") or "")
    return wrap_point_target(
        markdown,
        prompt_key=str(extra.get("prompt_key") or ""),
        translation=zh,
        is_negative=is_negative,
    )


def _block_extra_with_zh(block: BlockAnno, extra: dict[str, Any]) -> dict[str, Any]:
    zh = ""
    if block.extra:
        zh = str(block.extra.get("translation") or "")
    out = dict(extra)
    if zh:
        out["translation"] = zh
        out["mt_domain"] = str((block.extra or {}).get("mt_domain") or "")
    return out


def _emit_from_point(
    img: Image.Image,
    pt_x: float,
    pt_y: float,
    *,
    page_id: str,
    out_dir: Path,
    idx: int,
    target: str,
    region: str,
    is_negative: bool,
    extra: dict[str, Any],
    rng: random.Random | None = None,
) -> PointSample:
    w, h = img.size
    rng = rng or random.Random(idx)
    area_frac = _choose_area_frac(extra, rng)
    sample_extra = _frac_meta(extra, area_frac)
    marked = draw_crosshair(img, pt_x, pt_y, _spec_for(w, h, area_frac))
    tag = "neg" if is_negative else "p"
    name = f"{page_id}__{tag}{idx}.jpg"
    path = out_dir / name
    marked.save(path, format="JPEG", quality=92)
    nx, ny = pixel_to_norm(pt_x, pt_y, w, h)
    meta = {
        **sample_extra,
        "page_id": page_id,
        "point": [pt_x, pt_y],
        "point_norm": [nx, ny],
        "image_w": w,
        "image_h": h,
        "region": region,
        "is_negative": is_negative,
    }
    if is_negative:
        meta["block_id"] = None
    return PointSample(
        sample_id=f"{page_id}:{tag}{idx}",
        image_path=str(path.resolve()),
        task="POINT",
        target=_wrap_md(target, sample_extra, is_negative=is_negative),
        meta=meta,
    )


def emit_pool_samples(
    img: Image.Image,
    blocks: list[BlockAnno],
    *,
    pool_id: str,
    page_id: str,
    marked_dir: Path,
    seed: int,
    template_stem: str,
    prompt_key: str,
    layout_name: str | None = None,
    capture_meta: dict[str, Any] | None = None,
    page_html: str | None = None,
    chrome_bands: list[dict[str, Any]] | None = None,
) -> list[PointSample]:
    """Dispatch one rendered page into pool-specific POINT samples."""
    rng = random.Random(seed)
    w, h = img.size
    page_area = float(max(1, w * h))
    all_boxes = [r for b in blocks for r in b.ink_rects()]
    occ_boxes = occupancy_boxes(blocks, page_html)
    special_boxes = [r for b in blocks if is_special_block(b) for r in b.ink_rects()]
    extra = _base_meta(
        pool_id=pool_id,
        template_stem=template_stem,
        prompt_key=prompt_key,
        layout_name=layout_name,
        capture_meta=capture_meta or {},
    )
    marked_dir.mkdir(parents=True, exist_ok=True)

    if pool_id == "core_inner":
        skip_ids = clustered_member_ids(blocks, page_html)
        kept = pick_spread(
            [b for b in large_blocks(blocks, page_area, image_w=w, image_h=h) if b.block_id not in skip_ids],
            5,
            rng,
        )
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=10,
            r_max=10,
            n_negatives=0,
            seed=seed,
            avoid_boxes=all_boxes,
            coverage="diamond5",
            marker="x45",
            meta_extra=extra,
            mix_marker_scales=bool(extra.get("mix_marker_scales")),
        )
        return _stamp([s for s in batch if not s.meta.get("is_negative")], extra)

    if pool_id == "empty_clear":
        batch = build_point_samples_for_page(
            img,
            [],
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=4,
            seed=seed,
            avoid_boxes=occ_boxes + special_boxes,
            coverage="inner_area",
            marker="x45",
            meta_extra=extra,
            neg_clearance=Q1_NEG_CLEARANCE_PX,
            mix_marker_scales=bool(extra.get("mix_marker_scales")),
        )
        samples = _stamp([s for s in batch if s.meta.get("is_negative")], extra)
        for j, hit in enumerate(_chrome_empty_points(chrome_bands or [], rng, n=1)):
            x, y, region = hit
            samples.append(
                _emit_from_point(
                    img,
                    x,
                    y,
                    page_id=page_id,
                    out_dir=marked_dir,
                    idx=80 + j,
                    target="",
                    region=region,
                    is_negative=True,
                    extra=extra,
                )
            )
        return samples

    if pool_id == "empty_special":
        specials = [
            b
            for b in blocks
            if is_special_block(b) and block_fully_in_frame(b, w, h)
        ]
        rng.shuffle(specials)
        samples: list[PointSample] = []
        idx = 0
        for b in specials[:4]:
            kind = classify_a2_kind(b.extra, (b.extra or {}).get("tag"), b.markdown or "")
            if kind not in {"table", "image", "chart"}:
                from point_ocr.filter_qa import bucket_for_block

                bucket = bucket_for_block(b.markdown or "", tag=(b.extra or {}).get("tag"))
                if bucket == "table":
                    kind = "table"
            box = (b.ink_rects() or [b.bbox])[0]
            pts = sample_points_in_block(
                box,
                n=2,
                coverage="inner_area",
                rng=rng,
                block_id=b.block_id,
                image_w=w,
                image_h=h,
            )
            for pt in pts:
                samples.append(
                    _emit_from_point(
                        img,
                        pt.x,
                        pt.y,
                        page_id=page_id,
                        out_dir=marked_dir,
                        idx=idx,
                        target="",
                        region=f"special_{kind or 'other'}",
                        is_negative=True,
                        extra={
                            **extra,
                            "a2_kind": kind,
                            "special_block_id": b.block_id,
                            "bbox": list(box.as_tuple()),
                        },
                    )
                )
                idx += 1
        return samples

    if pool_id == "empty_boundary":
        pts = sample_boundary_empty_points(w, h, occ_boxes + special_boxes, n=8, rng=rng)
        special_full = [r for b in blocks if is_special_block(b) for r in b.ink_rects()]
        samples = []
        for i, p in enumerate(pts):
            if any(box.contains(p.x, p.y) for box in special_full):
                continue
            if any(box.contains(p.x, p.y) for box in occ_boxes):
                continue
            samples.append(
                _emit_from_point(
                    img,
                    p.x,
                    p.y,
                    page_id=page_id,
                    out_dir=marked_dir,
                    idx=i,
                    target="",
                    region=p.region or "neg_boundary",
                    is_negative=True,
                    extra=extra,
                )
            )
        return samples

    if pool_id == "multi_frag":
        kept = multi_frag_blocks(blocks, image_w=w, image_h=h)[:5]
        if not kept:
            return []
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=10,
            r_max=10,
            n_negatives=0,
            seed=seed,
            avoid_boxes=all_boxes,
            coverage="diamond5",
            marker="x45",
            meta_extra=extra,
            mix_marker_scales=bool(extra.get("mix_marker_scales")),
        )
        return _stamp([s for s in batch if not s.meta.get("is_negative")], extra)

    if pool_id == "semantic_group":
        return _emit_semantic(
            img,
            blocks,
            page_html=page_html or "",
            page_id=page_id,
            marked_dir=marked_dir,
            seed=seed,
            extra=extra,
            rng=rng,
        )

    raise KeyError(f"unknown pool_id {pool_id!r}")


def _emit_semantic(
    img: Image.Image,
    blocks: list[BlockAnno],
    *,
    page_html: str,
    page_id: str,
    marked_dir: Path,
    seed: int,
    extra: dict[str, Any],
    rng: random.Random,
) -> list[PointSample]:
    w, h = img.size
    by_id = {b.block_id: b for b in blocks}
    groups = parse_semantic_groups(page_html)
    samples: list[PointSample] = []
    order = list(range(len(groups)))
    rng.shuffle(order)
    n_take = min(3, max(1, len(groups))) if groups else 0
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
        if not members or not all(block_fully_in_frame(mb, w, h) for mb in members):
            continue
        union = union_bbox([r for m in members for r in m.ink_rects()])
        if union is None:
            continue
        labeled = BlockAnno(
            block_id=src.block_id,
            bbox=union,
            markdown=g.markdown,
            rects=[union],
            extra={
                **(src.extra or {}),
                "a2_kind": "semantic_group",
                "group_id": g.group_id,
                "group_kind": g.kind,
                "group_role": "body" if pick_body else "seed",
                "translation": g.translation,
            },
        )
        group_extra = {
            **extra,
            "a2_kind": "semantic_group",
            "group_id": g.group_id,
            "group_kind": g.kind,
            "group_role": "body" if pick_body else "seed",
            "n_fragments": 1,
            "unit_bbox": list(union.as_tuple()),
            "translation": g.translation,
        }
        samples.extend(
            _emit_one_block(
                img,
                labeled,
                page_id=page_id,
                out_dir=marked_dir,
                idx=gi,
                extra=group_extra,
                seed=seed + 1000 + gi,
            )
        )

    def _is_trunc(b: BlockAnno) -> bool:
        ex = b.extra or {}
        if ex.get("truncate_neighbor") not in (None, "", "0", 0, False):
            return True
        if b.block_id in {"orphan_nav", "d_outside", "n_outside", "w_outside", "r_outside"}:
            return True
        if str(b.block_id).endswith("_outside") or str(b.block_id).endswith("outside"):
            return True
        return False

    member_ids = {mid for g in groups for mid in g.member_ids}
    trunc = [b for b in blocks if _is_trunc(b) and block_fully_in_frame(b, w, h)]
    if not trunc:
        trunc = [
            b
            for b in blocks
            if b.block_id not in member_ids
            and len((b.markdown or "").strip()) >= 12
            and block_fully_in_frame(b, w, h)
        ][:2]
    rng.shuffle(trunc)
    for ti, b in enumerate(trunc[:1]):
        samples.extend(
            _emit_one_block(
                img,
                b,
                page_id=page_id,
                out_dir=marked_dir,
                idx=80 + ti,
                extra={**extra, "a2_kind": "truncate_neighbor", "group_role": "outside", "truncate_neg": True},
                seed=seed + 2000 + ti,
            )
        )
    return samples


def _group_gap_point(
    members: list[BlockAnno],
    rng: random.Random,
) -> tuple[float, float] | None:
    """Point in the gutter between two members of the same short unit."""
    ordered = sorted(members, key=lambda m: (m.bbox.y0, m.bbox.x0, m.block_id))
    for a, b in zip(ordered, ordered[1:]):
        gap = facing_gap_px(a.bbox, b.bbox)
        if gap is None or gap < 4 or gap > 36:
            continue
        strip = gap_strip(a.bbox, b.bbox)
        if strip is None or strip.width() < 4 or strip.height() < 4:
            continue
        inset_x = min(2.0, max(0.5, strip.width() * 0.15))
        inset_y = min(2.0, max(0.5, strip.height() * 0.15))
        if strip.width() <= 2 * inset_x + 1 or strip.height() <= 2 * inset_y + 1:
            continue
        x = rng.uniform(strip.x0 + inset_x, strip.x1 - inset_x)
        y = rng.uniform(strip.y0 + inset_y, strip.y1 - inset_y)
        return x, y
    return None


def _chrome_empty_points(
    bands: list[dict[str, Any]],
    rng: random.Random,
    *,
    n: int = 2,
) -> list[tuple[float, float, str]]:
    usable: list[tuple[str, float, float, float, float]] = []
    for row in bands:
        bb = row.get("bbox") or []
        if len(bb) < 4:
            continue
        x0, y0, x1, y1 = map(float, bb[:4])
        if (x1 - x0) < 20 or (y1 - y0) < 10:
            continue
        usable.append((str(row.get("band") or "head"), x0, y0, x1, y1))
    if not usable:
        return []
    out: list[tuple[float, float, str]] = []
    for _ in range(n):
        band, x0, y0, x1, y1 = rng.choice(usable)
        x = rng.uniform(x0 + 4, x1 - 4)
        y = rng.uniform(y0 + 3, y1 - 3)
        out.append((x, y, f"chrome_{band}"))
    return out


def _emit_one_block(
    img: Image.Image,
    block: BlockAnno,
    *,
    page_id: str,
    out_dir: Path,
    idx: int,
    extra: dict[str, Any],
    seed: int,
    coverage: str = "diamond5",
) -> list[PointSample]:
    md = (block.markdown or "").strip()
    if not md:
        return []
    w, h = img.size
    frags = block.ink_rects()
    rng = random.Random(seed + idx)
    if coverage in {"diamond5", "diamond"}:
        box = frags[0] if len(frags) == 1 else (frags[0] if frags else block.bbox)
        if len(frags) > 1:
            pts = sample_points_in_fragments(
                frags,
                n=10,
                coverage="diamond5",
                rng=rng,
                block_id=block.block_id,
                image_w=w,
                image_h=h,
            )
        else:
            pts = sample_diamond5_points(
                box,
                rng=rng,
                block_id=block.block_id,
                image_w=w,
                image_h=h,
                fragment_index=0,
            )
    elif len(frags) > 1:
        pts = sample_points_in_fragments(
            frags, n=1, coverage=coverage, rng=rng, block_id=block.block_id, image_w=w, image_h=h
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
    samples: list[PointSample] = []
    for j, pt in enumerate(pts):
        block_extra = _block_extra_with_zh(block, extra)
        area_frac = _choose_area_frac(block_extra, rng)
        sample_extra = _frac_meta(block_extra, area_frac)
        marked = draw_crosshair(img, pt.x, pt.y, _spec_for(w, h, area_frac))
        name = f"{page_id}__{block.block_id}__p{idx}_{j}.jpg"
        path = out_dir / name
        marked.save(path, format="JPEG", quality=92)
        nx, ny = pixel_to_norm(pt.x, pt.y, w, h)
        frag = pt.fragment_bbox or block.bbox
        meta = {
            **sample_extra,
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
        }
        if block.extra:
            for k in ("a2_kind", "group_id", "group_kind", "group_role", "visible_frac", "in_viewport_frac"):
                val = block.extra.get(k)
                if val is not None:
                    meta[k] = val
        samples.append(
            PointSample(
                sample_id=f"{page_id}:{block.block_id}:p{idx}_{j}",
                image_path=str(path.resolve()),
                task="POINT",
                target=_wrap_md(block.markdown, sample_extra),
                meta=meta,
            )
        )
    return samples

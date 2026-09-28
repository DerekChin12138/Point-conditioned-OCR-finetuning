"""Build POINT samples: draw crosshair on page image + emit dataset records."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from point_ocr.dataset_format import PointSample
from point_ocr.marker import (
    CURRENT_MARKER_TAG,
    MARKER_AREA_FRAC,
    MARKER_AREA_FRAC_LEGACY,
    MARKER_AREA_SCALE_LARGE,
    MARKER_AREA_SCALE_SMALL,
    area_frac_for_scale,
    MARKER_SPEC,
    MARKER_SPEC_A2,
    MARKER_SPEC_V2,
    draw_crosshair,
    draw_scatter_star,
    draw_translucent_dot,
    marker_extent_ok,
    sample_marker_area_frac,
    scale_dot_spec_for_image,
    spec_for_image,
)
from point_ocr.ocr_mt import wrap_point_target
from point_ocr.prompts import pixel_to_norm
from point_ocr.sample_points import (
    BBox,
    DEFAULT_NEG_CLEARANCE_PX,
    points_per_block_range,
    sample_negative_points,
    sample_points_in_fragments,
)


@dataclass
class BlockAnno:
    block_id: str
    bbox: BBox
    markdown: str
    extra: dict[str, Any] | None = None
    rects: list[BBox] = field(default_factory=list)

    def ink_rects(self) -> list[BBox]:
        """Fragment boxes for sampling / negatives; fall back to union bbox."""
        if self.rects:
            return list(self.rects)
        return [self.bbox]

    def ink_area(self) -> float:
        return sum(r.area() for r in self.ink_rects())


def build_point_samples_for_page(
    page_image: Image.Image,
    blocks: list[BlockAnno],
    *,
    page_id: str,
    out_image_dir: Path,
    r_min: int = 2,
    r_max: int = 5,
    n_negatives: int = 4,
    seed: int = 0,
    jpeg_quality: int = 92,
    avoid_boxes: list[BBox] | None = None,
    coverage: str = "center_band",
    band_half_frac: float = 0.35,
    marker: str = "x45",
    allow_empty_target: bool = False,
        meta_extra: dict[str, Any] | None = None,
        neg_clearance: float | None = None,
        mix_marker_scales: bool = False,
        area_scale: float | None = None,
        area_frac: float | None = None,
    ) -> list[PointSample]:
    """From one page + block annotations, create multi-point POINT samples.

    Writes marked images under `out_image_dir` and returns PointSample list.

    ``blocks`` are used for *positive* targets. ``avoid_boxes`` (default: all
    ink *fragments* of ``blocks``) are forbidden for negatives — pass *all*
    on-page text fragment boxes even if some blocks were filtered out of
    positives, otherwise a negative can land on visible unlabeled text.

    ``marker``: ``\"x45\"``/``\"a1\"`` current magenta X (45° cross);
    ``\"v2\"``/``\"dot\"`` adaptive circle (legacy); ``\"a2\"`` scatter star (legacy).
    ``allow_empty_target``: keep blocks whose markdown is empty (deny).
    """
    m = marker.strip().lower()
    if m in {"v2", "dot", "circle", "a1_v2", "a2_v2"}:
        draw_fn = draw_translucent_dot
        w0, h0 = page_image.size
        spec = scale_dot_spec_for_image(w0, h0, MARKER_SPEC_V2)
        marker_tag = "v2"
    elif m in {"a2", "scatter", "scatter_star"}:
        draw_fn = draw_scatter_star
        spec = MARKER_SPEC_A2
        marker_tag = "a2"
    else:
        draw_fn = draw_crosshair
        spec = spec_for_image(*page_image.size)
        marker_tag = CURRENT_MARKER_TAG
    if not marker_extent_ok(*page_image.size, spec):
        raise ValueError(f"Image too small for marker: {page_image.size}")

    rng = random.Random(seed)
    out_image_dir.mkdir(parents=True, exist_ok=True)
    w, h = page_image.size
    samples: list[PointSample] = []
    base_meta = dict(meta_extra or {})
    base_meta.setdefault("marker", marker_tag)
    prompt_key = str(base_meta.get("prompt_key") or "")

    def _x_spec_and_frac() -> tuple[Any, float]:
        """Pick the marker geometry and page-area fraction for one marked image.

        Current protocol: uniform random in [MARKER_AREA_FRAC_MIN, MAX] per image.
        ``area_frac`` / ``area_scale`` force a fixed size (legacy / GRPO_2 remakes).
        """
        if draw_fn is not draw_crosshair:
            return spec, float(MARKER_AREA_FRAC)
        if area_frac is not None:
            frac = float(area_frac)
        elif area_scale is not None:
            frac = area_frac_for_scale(float(area_scale))
        else:
            frac = sample_marker_area_frac(rng)
        return spec_for_image(w, h, area_frac=frac), frac

    if avoid_boxes is not None:
        neg_avoid = list(avoid_boxes)
    else:
        neg_avoid = [r for b in blocks for r in b.ink_rects()]

    for block in blocks:
        if not block.markdown.strip() and not allow_empty_target:
            continue
        frags = block.ink_rects()
        r = points_per_block_range(r_min, r_max, rng=rng)
        pts = sample_points_in_fragments(
            frags,
            n=r,
            rng=rng,
            block_id=block.block_id,
            coverage=coverage,
            band_half_frac=band_half_frac,
            image_w=w,
            image_h=h,
        )
        if not pts:
            continue
        all_rects = [list(f.as_tuple()) for f in frags]
        zh = str((block.extra or {}).get("translation") or "")
        for i, pt in enumerate(pts):
            local_spec, sample_frac = _x_spec_and_frac()
            marked = draw_fn(page_image, pt.x, pt.y, local_spec)
            name = f"{page_id}__{block.block_id}__p{i}.jpg"
            path = out_image_dir / name
            marked.save(path, format="JPEG", quality=jpeg_quality)
            nx, ny = pixel_to_norm(pt.x, pt.y, w, h)
            frag_box = pt.fragment_bbox or (
                frags[pt.fragment_index] if pt.fragment_index is not None and pt.fragment_index < len(frags) else block.bbox
            )
            meta = {
                **base_meta,
                "page_id": page_id,
                "block_id": block.block_id,
                "point": [pt.x, pt.y],
                "point_norm": [nx, ny],
                "image_w": w,
                "image_h": h,
                "region": pt.region,
                # Point-local fragment (green debug box should wrap ink under crosshair)
                "bbox": list(frag_box.as_tuple()),
                "bboxes": all_rects,
                "bbox_union": list(block.bbox.as_tuple()),
                "n_fragments": len(frags),
                "fragment_index": pt.fragment_index,
                "is_negative": False,
                "marker_area_frac": float(sample_frac),
                "marker_area_scale": float(sample_frac) / MARKER_AREA_FRAC_LEGACY,
            }
            if block.extra:
                for k in (
                    "a2_kind",
                    "a2_slice",
                    "group_id",
                    "group_role",
                    "visible_frac",
                    "in_viewport_frac",
                    "translation",
                    "mt_domain",
                ):
                    if k in block.extra:
                        meta[k] = block.extra[k]
            samples.append(
                PointSample(
                    sample_id=f"{page_id}:{block.block_id}:p{i}",
                    image_path=str(path.resolve()),
                    task="POINT",
                    target=wrap_point_target(
                        block.markdown,
                        prompt_key=prompt_key,
                        translation=zh,
                        is_negative=False,
                    ),
                    meta=meta,
                )
            )

    clearance = DEFAULT_NEG_CLEARANCE_PX if neg_clearance is None else float(neg_clearance)
    neg_pts = sample_negative_points(
        w, h, neg_avoid, n=n_negatives, rng=rng, clearance=clearance
    )
    for i, pt in enumerate(neg_pts):
        local_spec, sample_frac = _x_spec_and_frac()
        marked = draw_fn(page_image, pt.x, pt.y, local_spec)
        name = f"{page_id}__neg__n{i}.jpg"
        path = out_image_dir / name
        marked.save(path, format="JPEG", quality=jpeg_quality)
        nx, ny = pixel_to_norm(pt.x, pt.y, w, h)
        samples.append(
            PointSample(
                sample_id=f"{page_id}:neg:n{i}",
                image_path=str(path.resolve()),
                task="POINT",
                target="",
                meta={
                    **base_meta,
                    "page_id": page_id,
                    "block_id": None,
                    "point": [pt.x, pt.y],
                    "point_norm": [nx, ny],
                    "image_w": w,
                    "image_h": h,
                    "region": pt.region,
                    "is_negative": True,
                    "marker_area_frac": float(sample_frac),
                    "marker_area_scale": float(sample_frac) / MARKER_AREA_FRAC_LEGACY,
                },
            )
        )

    return samples


def load_blocks_json(path: Path) -> list[BlockAnno]:
    """Load blocks from JSON list of {id, bbox:[x0,y0,x1,y1], markdown, rects?}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[BlockAnno] = []
    for row in data:
        x0, y0, x1, y1 = row["bbox"]
        rects: list[BBox] = []
        for rr in row.get("rects") or []:
            if not rr or len(rr) < 4:
                continue
            a, b, c, d = map(float, rr[:4])
            if c - a >= 2 and d - b >= 2:
                rects.append(BBox(a, b, c, d))
        out.append(
            BlockAnno(
                block_id=str(row["id"]),
                bbox=BBox(float(x0), float(y0), float(x1), float(y1)),
                markdown=str(row.get("markdown", "")),
                rects=rects,
                extra={k: v for k, v in row.items() if k not in {"id", "bbox", "markdown", "rects"}},
            )
        )
    return out

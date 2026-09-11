"""Build POINT samples: draw crosshair on page image + emit dataset records."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from point_ocr.dataset_format import PointSample
from point_ocr.marker import MARKER_SPEC, draw_crosshair, marker_extent_ok
from point_ocr.sample_points import (
    BBox,
    points_per_block_range,
    sample_negative_points,
    sample_points_in_block,
)


@dataclass
class BlockAnno:
    block_id: str
    bbox: BBox
    markdown: str
    extra: dict[str, Any] | None = None


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
) -> list[PointSample]:
    """From one page + block annotations, create multi-point POINT samples.

    Writes marked images under `out_image_dir` and returns PointSample list.
    """
    if not marker_extent_ok(*page_image.size):
        raise ValueError(f"Image too small for marker: {page_image.size}")

    rng = random.Random(seed)
    out_image_dir.mkdir(parents=True, exist_ok=True)
    w, h = page_image.size
    samples: list[PointSample] = []

    boxes = [b.bbox for b in blocks]

    for block in blocks:
        if not block.markdown.strip():
            continue
        r = points_per_block_range(r_min, r_max, rng=rng)
        pts = sample_points_in_block(
            block.bbox, n=r, rng=rng, block_id=block.block_id
        )
        for i, pt in enumerate(pts):
            marked = draw_crosshair(page_image, pt.x, pt.y, MARKER_SPEC)
            name = f"{page_id}__{block.block_id}__p{i}.jpg"
            path = out_image_dir / name
            marked.save(path, format="JPEG", quality=jpeg_quality)
            samples.append(
                PointSample(
                    sample_id=f"{page_id}:{block.block_id}:p{i}",
                    image_path=str(path.resolve()),
                    task="POINT",
                    target=block.markdown,
                    meta={
                        "page_id": page_id,
                        "block_id": block.block_id,
                        "point": [pt.x, pt.y],
                        "region": pt.region,
                        "bbox": list(block.bbox.as_tuple()),
                        "is_negative": False,
                    },
                )
            )

    neg_pts = sample_negative_points(w, h, boxes, n=n_negatives, rng=rng)
    for i, pt in enumerate(neg_pts):
        marked = draw_crosshair(page_image, pt.x, pt.y, MARKER_SPEC)
        name = f"{page_id}__neg__n{i}.jpg"
        path = out_image_dir / name
        marked.save(path, format="JPEG", quality=jpeg_quality)
        samples.append(
            PointSample(
                sample_id=f"{page_id}:neg:n{i}",
                image_path=str(path.resolve()),
                task="POINT",
                target="",
                meta={
                    "page_id": page_id,
                    "block_id": None,
                    "point": [pt.x, pt.y],
                    "region": pt.region,
                    "is_negative": True,
                },
            )
        )

    return samples


def load_blocks_json(path: Path) -> list[BlockAnno]:
    """Load blocks from JSON list of {id, bbox:[x0,y0,x1,y1], markdown}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[BlockAnno] = []
    for row in data:
        x0, y0, x1, y1 = row["bbox"]
        out.append(
            BlockAnno(
                block_id=str(row["id"]),
                bbox=BBox(float(x0), float(y0), float(x1), float(y1)),
                markdown=str(row.get("markdown", "")),
                extra={k: v for k, v in row.items() if k not in {"id", "bbox", "markdown"}},
            )
        )
    return out

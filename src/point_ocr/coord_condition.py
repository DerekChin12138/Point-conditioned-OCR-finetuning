"""q1_coordinate: unmarked page + pixel point_2d (top-left origin). No visual X."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from point_ocr.prompts import format_coord_prompt_q1, pixel_to_xy_int


def render_path_for_meta(meta: dict[str, Any], *, pools_root: Path) -> Path:
    """Map a Q1 ShareGPT metadata row to the unmarked render PNG."""
    pool = str(meta.get("pool_id") or "")
    page_id = str(meta.get("page_id") or "")
    if not pool or not page_id:
        raise ValueError("metadata missing pool_id/page_id")
    return Path(pools_root) / pool / "renders" / f"{page_id}.png"


def coord_sharegpt_row(
    *,
    sample_id: str,
    image_path: str,
    target: str,
    px: float,
    py: float,
    image_w: int,
    image_h: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """One ShareGPT record: unmarked PNG + point_2d pixels in the prompt."""
    prompt = format_coord_prompt_q1(px, py, image_w, image_h)
    xi, yi = pixel_to_xy_int(px, py, image_w, image_h)
    out_meta = {
        **meta,
        "sample_id": sample_id,
        "task": "POINT",
        "prompt_key": "coord_q1",
        "marker": "none",
        "point": [px, py],
        "point_2d": [xi, yi],
        "coord_origin": "top-left",
        "coord_unit": "pixel",
        "image_w": image_w,
        "image_h": image_h,
    }
    return {
        "messages": [
            {"role": "user", "content": f"<image>{prompt}"},
            {"role": "assistant", "content": target},
        ],
        "images": [image_path],
        "metadata": out_meta,
    }

"""Playwright render of HTML pages + DOM bbox extraction for synth data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from point_ocr.html_to_md import extract_blocks_from_page_html, html_fragment_to_markdown


# Prefer *text ink* bounds over full element box.
# Block-level elements (h1/p) are often width:100%, so getBoundingClientRect()
# is far wider than the glyphs — that caused crosshairs beside short left-aligned text.
BBOX_JS = """
() => {
  const sx = window.scrollX || 0;
  const sy = window.scrollY || 0;

  function tightRect(el) {
    const er = el.getBoundingClientRect();
    let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
    let mode = 'element';

    try {
      const range = document.createRange();
      range.selectNodeContents(el);
      const rects = Array.from(range.getClientRects()).filter(
        (r) => r.width > 0.5 && r.height > 0.5
      );
      if (rects.length > 0) {
        for (const r of rects) {
          left = Math.min(left, r.left);
          top = Math.min(top, r.top);
          right = Math.max(right, r.right);
          bottom = Math.max(bottom, r.bottom);
        }
        // Clamp to element box (avoid absolute/overflow weirdness)
        left = Math.max(left, er.left);
        top = Math.max(top, er.top);
        right = Math.min(right, er.right);
        bottom = Math.min(bottom, er.bottom);
        mode = 'text';
      }
    } catch (e) {
      /* fall through */
    }

    if (!(right > left && bottom > top)) {
      left = er.left; top = er.top; right = er.right; bottom = er.bottom;
      mode = 'element';
    }

    // Tiny pad so the crosshair center sits on ink, not on the outer edge
    const pad = 1;
    return {
      left: left + pad,
      top: top + pad,
      right: right - pad,
      bottom: bottom - pad,
      mode,
      element: [er.left, er.top, er.right, er.bottom],
    };
  }

  return Array.from(document.querySelectorAll('[data-block-id]')).map((el) => {
    const t = tightRect(el);
    return {
      id: el.getAttribute('data-block-id'),
      bbox: [t.left + sx, t.top + sy, t.right + sx, t.bottom + sy],
      element_bbox: [
        t.element[0] + sx, t.element[1] + sy,
        t.element[2] + sx, t.element[3] + sy,
      ],
      bbox_mode: t.mode,
      tag: el.tagName.toLowerCase(),
      html: el.outerHTML,
    };
  });
}
"""


async def render_html_file(
    html_path: Path,
    out_dir: Path,
    *,
    viewport: tuple[int, int] = (1440, 900),
    device_scale_factor: float = 2.0,
    page_id: str | None = None,
    full_page: bool = True,
) -> dict[str, Any]:
    """Render one HTML file to PNG + block annotations (source-of-truth).

    Requires: pip install playwright && playwright install chromium

    Chromium is the headless browser engine Playwright uses to (1) paint HTML/CSS
    into a real screenshot and (2) run DOM JS to read text-tight bounding boxes.
    It is not used for model training itself.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise ImportError(
            "playwright is required for synth rendering. "
            "Install with: pip install '.[synth]' && playwright install chromium"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    page_id = page_id or html_path.stem
    image_path = out_dir / f"{page_id}.png"
    blocks_path = out_dir / f"{page_id}.blocks.json"
    meta_path = out_dir / f"{page_id}.meta.json"

    html_text = html_path.read_text(encoding="utf-8")
    # Also parse markdown targets from source HTML (not from OCR)
    source_blocks = {b["id"]: b for b in extract_blocks_from_page_html(html_text)}

    # Desktop templates opt into fixed viewport crop via meta comment
    if "data-capture=\"viewport\"" in html_text or 'data-capture="viewport"' in html_text:
        full_page = False

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]},
            device_scale_factor=device_scale_factor,
        )
        page = await context.new_page()
        await page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        await page.screenshot(path=str(image_path), full_page=full_page, type="png")
        dom_blocks = await page.evaluate(BBOX_JS)
        await browser.close()

    # Scale factor: getBoundingClientRect is in CSS pixels; screenshot may be denser
    # Playwright screenshot with device_scale_factor multiplies pixel size.
    scale = device_scale_factor
    annos: list[dict[str, Any]] = []
    for row in dom_blocks:
        bid = row["id"]
        x0, y0, x1, y1 = row["bbox"]
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        md = source_blocks.get(bid, {}).get("markdown")
        if md is None:
            md = html_fragment_to_markdown(row.get("html", ""))
        eb = row.get("element_bbox") or row["bbox"]
        annos.append(
            {
                "id": bid,
                "bbox": [x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                "element_bbox": [eb[0] * scale, eb[1] * scale, eb[2] * scale, eb[3] * scale],
                "bbox_mode": row.get("bbox_mode", "element"),
                "tag": row.get("tag"),
                "markdown": md,
                "html": row.get("html"),
            }
        )

    blocks_path.write_text(json.dumps(annos, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "page_id": page_id,
        "html_path": str(html_path),
        "image_path": str(image_path),
        "blocks_path": str(blocks_path),
        "viewport": list(viewport),
        "device_scale_factor": device_scale_factor,
        "n_blocks": len(annos),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def render_html_file_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
    import asyncio

    return asyncio.run(render_html_file(*args, **kwargs))

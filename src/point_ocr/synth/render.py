"""Playwright render of HTML pages + DOM bbox extraction for synth data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from point_ocr.html_to_md import extract_blocks_from_page_html, html_fragment_to_markdown

# Repo root: src/point_ocr/synth/render.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_KATEX_DIR = _REPO_ROOT / "data" / "synth" / "static" / "katex"

_KATEX_RENDER_JS = r"""
() => {
  function decodeEntities(s) {
    const ta = document.createElement('textarea');
    ta.innerHTML = s;
    return ta.value;
  }
  function normalizeTex(s) {
    s = decodeEntities(s || '');
    // "\\frac" (double-escaped in HTML) → "\frac"
    if (/\\\\[a-zA-Z_]/.test(s)) {
      s = s.replace(/\\\\/g, '\\');
    }
    return s;
  }
  if (typeof katex === 'undefined') return false;
  document.querySelectorAll('[data-latex]').forEach((el) => {
    const src = normalizeTex(el.getAttribute('data-latex') || '');
    try {
      katex.render(src, el, { throwOnError: false, displayMode: true });
    } catch (e) { /* leave original */ }
  });
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
        { left: '$', right: '$', display: false },
      ],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      throwOnError: false,
    });
  }
  return true;
}
"""


def _needs_katex(html_text: str) -> bool:
    if "katex.min.js" in html_text:
        return False
    return (
        "data-latex" in html_text
        or "\\(" in html_text
        or "$$" in html_text
        or "\\[" in html_text
    )


def _inject_katex(html_text: str) -> str:
    """Inject local KaTeX assets so formulas paint as math, not raw TeX source."""
    if not _needs_katex(html_text):
        return html_text
    if not (_KATEX_DIR / "katex.min.js").is_file():
        return html_text
    css = (_KATEX_DIR / "katex.min.css").resolve().as_uri()
    js = (_KATEX_DIR / "katex.min.js").resolve().as_uri()
    auto = (_KATEX_DIR / "auto-render.min.js").resolve().as_uri()
    head = (
        f'<link rel="stylesheet" href="{css}" />\n'
        f'<script src="{js}"></script>\n'
        f'<script src="{auto}"></script>\n'
    )
    if "</head>" in html_text:
        return html_text.replace("</head>", head + "</head>", 1)
    return head + html_text


def _prepare_html_for_render(html_path: Path) -> Path:
    """Return path to open in Chromium (may write a sibling .katex.html)."""
    text = html_path.read_text(encoding="utf-8")
    patched = _inject_katex(text)
    if patched == text:
        return html_path
    out = html_path.with_name(html_path.stem + ".katex.html")
    out.write_text(patched, encoding="utf-8")
    return out


# Prefer *text ink* bounds over full element box.
# Block-level elements (h1/p) are often width:100%, so getBoundingClientRect()
# is far wider than the glyphs — that caused crosshairs beside short left-aligned text.
BBOX_JS = """
() => {
  const sx = window.scrollX || 0;
  const sy = window.scrollY || 0;

  // Merge nearby line boxes; keep separate islands for column/flow jumps.
  function clusterFragments(raw) {
    if (!raw.length) return [];
    const rects = raw.map((r) => ({
      left: r.left, top: r.top, right: r.right, bottom: r.bottom, height: r.height,
    }));
    const heights = rects.map((r) => r.height).sort((a, b) => a - b);
    const medH = heights[Math.floor(heights.length / 2)] || 12;
    const pad = Math.max(6, medH * 0.6);
    const parent = rects.map((_, i) => i);
    const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    const unite = (a, b) => {
      a = find(a); b = find(b);
      if (a !== b) parent[b] = a;
    };
    const near = (a, b) => {
      const al = a.left - pad, at = a.top - pad, ar = a.right + pad, ab = a.bottom + pad;
      const bl = b.left - pad, bt = b.top - pad, br = b.right + pad, bb = b.bottom + pad;
      return al < br && ar > bl && at < bb && ab > bt;
    };
    for (let i = 0; i < rects.length; i++) {
      for (let j = i + 1; j < rects.length; j++) {
        if (near(rects[i], rects[j])) unite(i, j);
      }
    }
    const groups = {};
    for (let i = 0; i < rects.length; i++) {
      const root = find(i);
      (groups[root] || (groups[root] = [])).push(rects[i]);
    }
    return Object.values(groups).map((g) => {
      let L = Infinity, T = Infinity, R = -Infinity, B = -Infinity;
      for (const x of g) {
        L = Math.min(L, x.left); T = Math.min(T, x.top);
        R = Math.max(R, x.right); B = Math.max(B, x.bottom);
      }
      return { left: L, top: T, right: R, bottom: B };
    }).filter((f) => f.right - f.left > 1 && f.bottom - f.top > 1);
  }

  function tightRect(el) {
    const er = el.getBoundingClientRect();
    let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
    let mode = 'element';
    let fragments = [];

    try {
      const range = document.createRange();
      range.selectNodeContents(el);
      const lineRects = Array.from(range.getClientRects()).filter(
        (r) => r.width > 0.5 && r.height > 0.5
      );
      if (lineRects.length > 0) {
        fragments = clusterFragments(lineRects);
        for (const f of fragments) {
          left = Math.min(left, f.left);
          top = Math.min(top, f.top);
          right = Math.max(right, f.right);
          bottom = Math.max(bottom, f.bottom);
        }
        // Clamp to element box (avoid absolute/overflow weirdness)
        left = Math.max(left, er.left);
        top = Math.max(top, er.top);
        right = Math.min(right, er.right);
        bottom = Math.min(bottom, er.bottom);
        fragments = fragments.map((f) => ({
          left: Math.max(f.left, er.left),
          top: Math.max(f.top, er.top),
          right: Math.min(f.right, er.right),
          bottom: Math.min(f.bottom, er.bottom),
        })).filter((f) => f.right - f.left > 1 && f.bottom - f.top > 1);
        mode = fragments.length > 1 ? 'text_multi' : 'text';
      }
    } catch (e) {
      /* fall through */
    }

    if (!(right > left && bottom > top)) {
      left = er.left; top = er.top; right = er.right; bottom = er.bottom;
      fragments = [{ left, top, right, bottom }];
      mode = 'element';
    } else if (!fragments.length) {
      fragments = [{ left, top, right, bottom }];
    }

    // Tiny pad so the crosshair center sits on ink, not on the outer edge
    const pad = 1;
    const shrink = (f) => ({
      left: f.left + pad,
      top: f.top + pad,
      right: f.right - pad,
      bottom: f.bottom - pad,
    });
    const union = shrink({ left, top, right, bottom });
    const fr = fragments.map(shrink).filter((f) => f.right > f.left && f.bottom > f.top);
    return {
      left: union.left,
      top: union.top,
      right: union.right,
      bottom: union.bottom,
      fragments: fr.length ? fr : [union],
      mode,
      element: [er.left, er.top, er.right, er.bottom],
    };
  }

    // Fraction of a client-rect grid that is both inside the viewport and
    // still hits `el` (or a descendant). Out-of-viewport samples count as
    // misses so clipped (window-cropped) blocks get a low score.
    function visibleFrac(el, left, top, right, bottom) {
      const w = right - left, h = bottom - top;
      if (w < 2 || h < 2) return 0;
      const n = 5;
      const iw = window.innerWidth, ih = window.innerHeight;
      let hit = 0, total = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const x = left + ((i + 0.5) / n) * w;
          const y = top + ((j + 0.5) / n) * h;
          total += 1;
          if (x < 0 || y < 0 || x >= iw || y >= ih) {
            continue; // clipped by window → miss
          }
          const topEl = document.elementFromPoint(x, y);
          if (topEl && (el === topEl || el.contains(topEl))) hit += 1;
        }
      }
      return total ? hit / total : 0;
    }

    // Fraction of the tight bbox area that intersects the viewport.
    function inViewportFrac(left, top, right, bottom) {
      const iw = window.innerWidth, ih = window.innerHeight;
      const x0 = Math.max(left, 0), y0 = Math.max(top, 0);
      const x1 = Math.min(right, iw), y1 = Math.min(bottom, ih);
      const area = Math.max(0, right - left) * Math.max(0, bottom - top);
      if (area < 1) return 0;
      const vis = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
      return vis / area;
    }

  return Array.from(document.querySelectorAll('[data-block-id]')).map((el) => {
    const t = tightRect(el);
    const groupEl = el.closest('[data-semantic-group]');
    const winEl = el.closest('[data-window]');
    const vfBoxes = (t.fragments && t.fragments.length)
      ? t.fragments
      : [{ left: t.left, top: t.top, right: t.right, bottom: t.bottom }];
    // Score ink fragments, not the union bbox (column gutters are not misses).
    const vf = vfBoxes.reduce((s, f) => s + visibleFrac(el, f.left, f.top, f.right, f.bottom), 0)
      / vfBoxes.length;
    const ivf = Math.min(...vfBoxes.map((f) => inViewportFrac(f.left, f.top, f.right, f.bottom)));
    return {
      id: el.getAttribute('data-block-id'),
      bbox: [t.left + sx, t.top + sy, t.right + sx, t.bottom + sy],
      rects: t.fragments.map((f) => [
        f.left + sx, f.top + sy, f.right + sx, f.bottom + sy,
      ]),
      element_bbox: [
        t.element[0] + sx, t.element[1] + sy,
        t.element[2] + sx, t.element[3] + sy,
      ],
      bbox_mode: t.mode,
      tag: el.tagName.toLowerCase(),
      html: el.outerHTML,
      role: el.getAttribute('data-role'),
      a2_kind: el.getAttribute('data-a2-kind'),
      'data-latex': el.getAttribute('data-latex'),
      'data-mt-zh': el.getAttribute('data-mt-zh'),
      'data-mt-domain': el.getAttribute('data-mt-domain'),
      truncate_neighbor: el.getAttribute('data-truncate-neighbor'),
      semantic_group: groupEl ? groupEl.getAttribute('data-semantic-group') : null,
      group_kind: groupEl ? groupEl.getAttribute('data-group-kind') : null,
      visible_frac: vf,
      in_viewport_frac: ivf,
      window_id: winEl ? winEl.getAttribute('data-window') : null,
      window_focused: !!(winEl && winEl.getAttribute('data-focused') === 'true'),
    };
  });
}
"""


CHROME_BAND_JS = """
() => {
  const sx = window.scrollX || 0;
  const sy = window.scrollY || 0;
  return Array.from(document.querySelectorAll('[data-chrome-band]')).map((el) => {
    const r = el.getBoundingClientRect();
    return {
      band: el.getAttribute('data-chrome-band'),
      bbox: [r.left + sx, r.top + sy, r.right + sx, r.bottom + sy],
    };
  });
}
"""


def _parse_render_opts(
    html_text: str,
    *,
    viewport: tuple[int, int],
    full_page: bool,
) -> tuple[tuple[int, int], bool]:
    vp = viewport
    m = re.search(r'data-viewport=["\'](\d+)x(\d+)["\']', html_text)
    if m:
        vp = (int(m.group(1)), int(m.group(2)))
    if 'data-capture="viewport"' in html_text or "data-capture='viewport'" in html_text:
        full_page = False
    return vp, full_page


def _annos_from_dom(
    dom_blocks: list[dict[str, Any]],
    source_blocks: dict[str, dict[str, Any]],
    *,
    scale: float,
) -> list[dict[str, Any]]:
    annos: list[dict[str, Any]] = []
    for row in dom_blocks:
        bid = row["id"]
        x0, y0, x1, y1 = row["bbox"]
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        md = source_blocks.get(bid, {}).get("markdown")
        if md is None:
            md = html_fragment_to_markdown(row.get("html", ""))
        zh = source_blocks.get(bid, {}).get("translation") or row.get("data-mt-zh") or ""
        domain = source_blocks.get(bid, {}).get("mt_domain") or row.get("data-mt-domain") or ""
        eb = row.get("element_bbox") or row["bbox"]
        raw_rects = row.get("rects") or [row["bbox"]]
        rects = []
        for rr in raw_rects:
            if not rr or len(rr) < 4:
                continue
            a, b, c, d = (float(v) * scale for v in rr[:4])
            if c - a >= 2 and d - b >= 2:
                rects.append([a, b, c, d])
        if not rects:
            rects = [[x0 * scale, y0 * scale, x1 * scale, y1 * scale]]
        annos.append(
            {
                "id": bid,
                "bbox": [x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                "rects": rects,
                "element_bbox": [eb[0] * scale, eb[1] * scale, eb[2] * scale, eb[3] * scale],
                "bbox_mode": row.get("bbox_mode", "element"),
                "tag": row.get("tag"),
                "markdown": md,
                "translation": zh,
                "mt_domain": domain,
                "html": row.get("html"),
                "role": row.get("role"),
                "a2_kind": row.get("a2_kind"),
                "data-latex": row.get("data-latex"),
                "truncate_neighbor": row.get("truncate_neighbor"),
                "semantic_group": row.get("semantic_group"),
                "group_kind": row.get("group_kind"),
                "visible_frac": row.get("visible_frac"),
                "in_viewport_frac": row.get("in_viewport_frac"),
                "window_id": row.get("window_id"),
                "window_focused": row.get("window_focused"),
            }
        )
    return annos


def _write_render_outputs(
    *,
    html_path: Path,
    image_path: Path,
    blocks_path: Path,
    meta_path: Path,
    page_id: str,
    vp: tuple[int, int],
    device_scale_factor: float,
    annos: list[dict[str, Any]],
    chrome_bands: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    blocks_path.write_text(json.dumps(annos, ensure_ascii=False, indent=2), encoding="utf-8")
    chrome_path = blocks_path.with_name(blocks_path.name.replace(".blocks.json", ".chrome.json"))
    bands = chrome_bands or []
    chrome_path.write_text(json.dumps(bands, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "page_id": page_id,
        "html_path": str(html_path),
        "image_path": str(image_path),
        "blocks_path": str(blocks_path),
        "chrome_path": str(chrome_path),
        "viewport": list(vp),
        "device_scale_factor": device_scale_factor,
        "n_blocks": len(annos),
        "n_chrome_bands": len(bands),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


async def _screenshot_blocks(
    browser: Any,
    html_path: Path,
    image_path: Path,
    *,
    vp: tuple[int, int],
    device_scale_factor: float,
    full_page: bool,
    wait_until: str = "load",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = await browser.new_context(
        viewport={"width": vp[0], "height": vp[1]},
        device_scale_factor=device_scale_factor,
    )
    try:
        page = await context.new_page()
        # Local file:// pages have no network; "load" is enough and much faster
        # than "networkidle" (which used to dominate synth wall time).
        await page.goto(html_path.resolve().as_uri(), wait_until=wait_until)
        # Typeset KaTeX formulas when assets were injected.
        try:
            await page.wait_for_function("() => typeof katex !== 'undefined'", timeout=8000)
            await page.evaluate(_KATEX_RENDER_JS)
            # Allow fonts to settle before screenshot
            await page.wait_for_timeout(50)
        except Exception:
            pass
        await page.screenshot(path=str(image_path), full_page=full_page, type="png")
        blocks = await page.evaluate(BBOX_JS)
        chrome = await page.evaluate(CHROME_BAND_JS)
        return blocks, chrome
    finally:
        await context.close()


async def render_html_file(
    html_path: Path,
    out_dir: Path,
    *,
    viewport: tuple[int, int] = (1920, 1080),
    device_scale_factor: float = 1.0,
    page_id: str | None = None,
    full_page: bool = True,
    browser: Any | None = None,
    wait_until: str = "load",
) -> dict[str, Any]:
    """Render one HTML file to PNG + block annotations (source-of-truth).

    Requires: pip install playwright && playwright install chromium

    Chromium is the headless browser engine Playwright uses to (1) paint HTML/CSS
    into a real screenshot and (2) run DOM JS to read text-tight bounding boxes.
    It is not used for model training itself.

    Pass a live Playwright `browser` to reuse Chromium across many pages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    page_id = page_id or html_path.stem
    image_path = out_dir / f"{page_id}.png"
    blocks_path = out_dir / f"{page_id}.blocks.json"
    meta_path = out_dir / f"{page_id}.meta.json"

    html_text = html_path.read_text(encoding="utf-8")
    source_blocks = {b["id"]: b for b in extract_blocks_from_page_html(html_text)}
    vp, full_page = _parse_render_opts(html_text, viewport=viewport, full_page=full_page)
    render_html_path = _prepare_html_for_render(html_path)

    own_browser = browser is None
    if own_browser:
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError(
                "playwright is required for synth rendering. "
                "Install with: pip install '.[synth]' && playwright install chromium"
            ) from e
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                dom_blocks, chrome_raw = await _screenshot_blocks(
                    browser,
                    render_html_path,
                    image_path,
                    vp=vp,
                    device_scale_factor=device_scale_factor,
                    full_page=full_page,
                    wait_until=wait_until,
                )
            finally:
                await browser.close()
    else:
        dom_blocks, chrome_raw = await _screenshot_blocks(
            browser,
            render_html_path,
            image_path,
            vp=vp,
            device_scale_factor=device_scale_factor,
            full_page=full_page,
            wait_until=wait_until,
        )

    annos = _annos_from_dom(dom_blocks, source_blocks, scale=device_scale_factor)
    chrome_bands = []
    for row in chrome_raw or []:
        bb = row.get("bbox") or []
        if len(bb) < 4:
            continue
        chrome_bands.append(
            {
                "band": row.get("band"),
                "bbox": [float(v) * device_scale_factor for v in bb[:4]],
            }
        )
    return _write_render_outputs(
        html_path=html_path,
        image_path=image_path,
        blocks_path=blocks_path,
        meta_path=meta_path,
        page_id=page_id,
        vp=vp,
        device_scale_factor=device_scale_factor,
        annos=annos,
        chrome_bands=chrome_bands,
    )


class HtmlRenderSession:
    """Keep one Chromium process alive across many HTML renders."""

    def __init__(self, *, wait_until: str = "load") -> None:
        self._loop: Any = None
        self._pw: Any = None
        self._browser: Any = None
        self.wait_until = wait_until

    def __enter__(self) -> HtmlRenderSession:
        import asyncio

        from playwright.async_api import async_playwright

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._pw = self._loop.run_until_complete(async_playwright().start())
        self._browser = self._loop.run_until_complete(self._pw.chromium.launch(headless=True))
        return self

    def render(self, html_path: Path, out_dir: Path, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("browser", None)
        kwargs.setdefault("wait_until", self.wait_until)
        return self._loop.run_until_complete(
            render_html_file(html_path, out_dir, browser=self._browser, **kwargs)
        )

    def render_many(
        self,
        jobs: list[tuple[Path, Path, str]],
        *,
        concurrency: int = 4,
    ) -> list[dict[str, Any]]:
        """Render many HTML pages concurrently on this browser (separate contexts)."""
        import asyncio

        async def _run() -> list[dict[str, Any]]:
            sem = asyncio.Semaphore(max(1, concurrency))

            async def one(html_path: Path, out_dir: Path, page_id: str) -> dict[str, Any]:
                async with sem:
                    return await render_html_file(
                        html_path,
                        out_dir,
                        browser=self._browser,
                        page_id=page_id,
                        wait_until=self.wait_until,
                    )

            return list(
                await asyncio.gather(*[one(h, o, p) for h, o, p in jobs])
            )

        return self._loop.run_until_complete(_run())

    def __exit__(self, *exc: Any) -> None:
        if self._loop is None:
            return
        if self._browser is not None:
            self._loop.run_until_complete(self._browser.close())
        if self._pw is not None:
            self._loop.run_until_complete(self._pw.stop())
        self._loop.close()
        self._loop = None


def render_html_file_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
    import asyncio

    return asyncio.run(render_html_file(*args, **kwargs))

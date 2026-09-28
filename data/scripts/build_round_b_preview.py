#!/usr/bin/env python3
"""Round B human preview: chrome as layer 3 + diamond5 points + opaque X.

Does NOT compose a training mix. ~24 pages, stratified none/browser/office.

  export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
  uv run python data/scripts/generate_chrome_content.py
  uv run python data/scripts/build_round_b_preview.py
  # open data/review_galleries/preview_round_b/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from apply_content_pack import fill_from_pool  # noqa: E402
from point_ocr.build_point import load_blocks_json  # noqa: E402
from point_ocr.chrome.wrap import (  # noqa: E402
    BROWSER_SKINS,
    OFFICE_SKINS,
    ChromeSpec,
    sample_chrome_spec,
    wrap_page_html,
)
from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.marker import CURRENT_MARKER_TAG, draw_crosshair, spec_for_image  # noqa: E402
from point_ocr.noise import apply_screen_noise  # noqa: E402
from point_ocr.pools.select import large_blocks  # noqa: E402
from point_ocr.sample_points import sample_diamond5_points  # noqa: E402
from point_ocr.synth.render import HtmlRenderSession  # noqa: E402
from point_ocr.synth.window_viewport import sample_window_capture  # noqa: E402

TEMPLATES = [
    "01_article_twocol.html",
    "04_zh_news_twocol.html",
    "08_github_readme_dark.html",
    "12_wikipedia_article.html",
    "14_magazine_3col.html",
]


def _load_json_pool(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pool") or data


def _load_chrome_pool(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"browser_scenes": [], "office_scenes": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "browser_scenes": data.get("browser_scenes") or [],
        "office_scenes": data.get("office_scenes") or [],
    }


def _forced_spec(kind: str, skin: str | None, rng: random.Random, chrome_pool: dict) -> ChromeSpec:
    if kind == "none":
        return ChromeSpec(family="none", skin="none")
    if kind == "browser":
        spec = sample_chrome_spec(
            rng, chrome_pool, family_weights={"none": 0, "browser": 1, "office": 0}
        )
        if skin:
            spec.skin = skin
            spec.theme = "dark" if "dark" in skin else "light"
        return spec
    spec = sample_chrome_spec(
        rng, chrome_pool, family_weights={"none": 0, "browser": 0, "office": 1}
    )
    if skin:
        spec.skin = skin
    spec.show_status = True
    return spec


def _plan(n_none: int, n_per_browser: int, n_per_office: int) -> list[tuple[str, str | None]]:
    rows: list[tuple[str, str | None]] = [("none", None)] * n_none
    for skin in BROWSER_SKINS:
        rows.extend(("browser", skin) for _ in range(n_per_browser))
    for skin in OFFICE_SKINS:
        rows.extend(("office", skin) for _ in range(n_per_office))
    return rows


def _chrome_empty_point(bands: list[dict], rng: random.Random) -> tuple[float, float, str] | None:
    usable = []
    for row in bands:
        bb = row.get("bbox") or []
        if len(bb) < 4:
            continue
        x0, y0, x1, y1 = map(float, bb[:4])
        if (x1 - x0) < 20 or (y1 - y0) < 10:
            continue
        usable.append((str(row.get("band") or "head"), x0, y0, x1, y1))
    if not usable:
        return None
    band, x0, y0, x1, y1 = rng.choice(usable)
    x = rng.uniform(x0 + 4, x1 - 4)
    y = rng.uniform(y0 + 3, y1 - 3)
    return x, y, f"chrome_{band}"


def _stamp(
    img: Image.Image,
    x: float,
    y: float,
    *,
    page_id: str,
    idx: int,
    target: str,
    extra: dict[str, Any],
    marked_dir: Path,
) -> PointSample:
    w, h = img.size
    spec = spec_for_image(w, h)
    marked = draw_crosshair(img, x, y, spec)
    name = f"{page_id}__p{idx}.jpg"
    path = marked_dir / name
    marked.save(path, format="JPEG", quality=92)
    meta = {
        **extra,
        "page_id": page_id,
        "point": [x, y],
        "image_w": w,
        "image_h": h,
        "marker": CURRENT_MARKER_TAG,
        "is_negative": not bool((target or "").strip()),
    }
    return PointSample(
        sample_id=f"{page_id}:p{idx}",
        image_path=str(path.resolve()),
        task="POINT",
        target=target,
        meta=meta,
    )


def _write_gallery(
    *,
    out: Path,
    pages: list[dict[str, Any]],
    samples: list[PointSample],
) -> None:
    if out.exists():
        shutil.rmtree(out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True)
    page_cards = []
    for i, pg in enumerate(pages):
        src = Path(pg["render"])
        dst = img_dir / f"page_{i:02d}_{pg['page_id']}.jpg"
        im = Image.open(src).convert("RGB")
        im.thumbnail((1100, 900))
        im.save(dst, quality=88)
        rel = dst.relative_to(out).as_posix()
        page_cards.append(
            f"""<figure class="page">
            <figcaption><code>{html.escape(pg['page_id'])}</code>
            · {html.escape(pg['skin'])} · {html.escape(pg['template'])}</figcaption>
            <a href="{html.escape(rel)}" target="_blank"><img src="{html.escape(rel)}" /></a>
            </figure>"""
        )

    sample_cards = []
    for i, s in enumerate(samples):
        src = Path(s.image_path)
        dst = img_dir / f"s_{i:03d}.jpg"
        im = Image.open(src).convert("RGB")
        # bbox overlay for review only
        meta = s.meta
        bbox = meta.get("bbox")
        if bbox and len(bbox) >= 4:
            overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(overlay)
            x0, y0, x1, y1 = map(int, bbox[:4])
            dr.rectangle([x0, y0, x1, y1], outline=(0, 200, 80, 220), width=3)
            im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        im.save(dst, quality=90)
        rel = dst.relative_to(out).as_posix()
        gt = s.target or "（空串）"
        neg = not (s.target or "").strip()
        sample_cards.append(
            f"""<section class="card">
            <div class="meta">
              <b>#{i}</b> <code>{html.escape(s.sample_id)}</code>
              <span class="pill {'neg' if neg else 'pos'}">{'壳上负例' if neg else '正例'}</span>
              <span class="pill">{html.escape(str(meta.get('chrome_skin')))}</span>
              <span class="pill">region={html.escape(str(meta.get('region')))}</span>
            </div>
            <div class="split">
              <a href="{html.escape(rel)}" target="_blank"><img src="{html.escape(rel)}" /></a>
              <pre>{html.escape(gt)}</pre>
            </div>
            </section>"""
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<title>Round B preview</title>
<style>
body {{ margin:0; font:14px/1.45 ui-sans-serif,system-ui; background:#1b1b1d; color:#ececec; }}
h1,h2 {{ font-weight:600; }}
header, section.block {{ padding:16px 20px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px; }}
figure.page {{ margin:0; background:#242427; padding:8px; }}
figure.page img {{ width:100%; height:auto; display:block; }}
.card {{ background:#242427; margin:12px 0; padding:10px; }}
.split {{ display:grid; grid-template-columns:1.4fr 1fr; gap:12px; }}
.split img {{ width:100%; height:auto; }}
pre {{ white-space:pre-wrap; background:#111; padding:10px; min-height:80px; }}
.pill {{ display:inline-block; background:#333; padding:1px 8px; margin-left:6px; font-size:12px; }}
.pill.pos {{ background:#1f6f4a; }} .pill.neg {{ background:#6f1f1f; }}
figcaption, .meta {{ color:#9a9aa0; font-size:12px; margin-bottom:6px; }}
@media (max-width:900px) {{ .split {{ grid-template-columns:1fr; }} }}
</style></head>
<body>
<header>
<h1>Round B 预览</h1>
<p>壳体是第三层：下面整页图约一半无壳。标记 α=255，面积 0.5%。绿框仅审查用。</p>
<p>页 {len(pages)} · 标记样本 {len(samples)} · marker={CURRENT_MARKER_TAG}</p>
</header>
<section class="block">
<h2>1. 整页（看壳体是否逼真；无十字）</h2>
<div class="grid">{''.join(page_cards)}</div>
</section>
<section class="block">
<h2>2. 打标样本（五区点 + 壳上负例）</h2>
{''.join(sample_cards)}
</section>
</body></html>"""
    (out / "index.html").write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/preview_round_b")
    ap.add_argument("--gallery", type=Path, default=ROOT / "data/review_galleries/preview_round_b")
    ap.add_argument("--content-pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--chrome-pool", type=Path, default=ROOT / "data/synth/content_pools/chrome_pool.json")
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--n-none", type=int, default=8)
    ap.add_argument("--n-per-browser-skin", type=int, default=2)
    ap.add_argument("--n-per-office-skin", type=int, default=4)
    args = ap.parse_args()
    pw = Path.home() / ".cache" / "ms-playwright"
    if pw.is_dir() and "cursor-sandbox-cache" in os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(pw)

    content_pool = _load_json_pool(args.content_pool)
    chrome_pool = _load_chrome_pool(args.chrome_pool)
    plan = _plan(args.n_none, args.n_per_browser_skin, args.n_per_office_skin)
    rng = random.Random(args.seed)

    filled_dir = args.out / "filled_html"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"
    for d in (filled_dir, render_dir, marked_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    pages_meta: list[dict[str, Any]] = []
    samples: list[PointSample] = []

    tmpl_root = ROOT / "data/synth/templates"
    with HtmlRenderSession(wait_until="load") as session:
        for i, (kind, skin) in enumerate(plan):
            tmpl = tmpl_root / TEMPLATES[i % len(TEMPLATES)]
            page_id = f"rb_{i:02d}_{kind}_{skin or 'none'}"
            local = random.Random(args.seed + i * 97)
            html_src = tmpl.read_text(encoding="utf-8")
            filled, layout = fill_from_pool(
                html_src, content_pool, local, extra_paragraphs=6, layout_set="dense"
            )
            spec = _forced_spec(kind, skin, local, chrome_pool)
            wrapped = wrap_page_html(filled, spec)
            html_path = filled_dir / f"{page_id}.html"
            html_path.write_text(wrapped, encoding="utf-8")
            cap = sample_window_capture(local)
            meta = session.render(
                html_path,
                render_dir,
                page_id=page_id,
                viewport=cap.viewport,
                device_scale_factor=1.0,
                full_page=False,
            )
            img = apply_screen_noise(
                Image.open(meta["image_path"]).convert("RGB"),
                rng=random.Random(args.seed + i * 31),
            )
            blocks = load_blocks_json(Path(meta["blocks_path"]))
            chrome_path = Path(meta.get("chrome_path") or "")
            bands = []
            if chrome_path.is_file():
                bands = json.loads(chrome_path.read_text(encoding="utf-8"))
            extra = {
                "template": tmpl.name,
                "layout": layout,
                "prompt_key": "a2_v2",
                **spec.to_meta(),
            }
            pages_meta.append(
                {
                    "page_id": page_id,
                    "render": meta["image_path"],
                    "skin": spec.skin,
                    "template": tmpl.name,
                }
            )
            w, h = img.size
            kept = large_blocks(blocks, float(w * h), image_w=w, image_h=h)
            if kept:
                block = local.choice(kept[:4])
                box = (block.ink_rects() or [block.bbox])[0]
                pts = sample_diamond5_points(
                    box, n_regions=3, points_per_region=2, rng=local, block_id=block.block_id
                )
                # gallery: one diamond + one corner if present
                shown = []
                dia = [p for p in pts if p.region == "diamond"]
                cor = [p for p in pts if p.region != "diamond"]
                if dia:
                    shown.append(dia[0])
                if cor:
                    shown.append(cor[0])
                if not shown:
                    shown = pts[:2]
                for j, p in enumerate(shown):
                    samples.append(
                        _stamp(
                            img,
                            p.x,
                            p.y,
                            page_id=page_id,
                            idx=j,
                            target=block.markdown,
                            extra={
                                **extra,
                                "block_id": block.block_id,
                                "region": p.region,
                                "bbox": list(box.as_tuple()),
                            },
                            marked_dir=marked_dir,
                        )
                    )
            if spec.family != "none":
                hit = _chrome_empty_point(bands, local)
                if hit:
                    x, y, region = hit
                    samples.append(
                        _stamp(
                            img,
                            x,
                            y,
                            page_id=page_id,
                            idx=90,
                            target="",
                            extra={**extra, "region": region, "bbox": None},
                            marked_dir=marked_dir,
                        )
                    )
            print(f"[{i+1}/{len(plan)}] {page_id} skin={spec.skin} blocks={len(blocks)}", flush=True)

    jsonl = args.out / "point_sharegpt.jsonl"
    write_jsonl(jsonl, samples)
    _write_gallery(out=args.gallery, pages=pages_meta, samples=samples)
    print(f"jsonl {jsonl} n={len(samples)}", flush=True)
    print(f"gallery {args.gallery / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

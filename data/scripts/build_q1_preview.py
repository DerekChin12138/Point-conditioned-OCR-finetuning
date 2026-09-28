#!/usr/bin/env python3
"""Q1 training-faithful preview: a2_v3 + occ∈[25%,70%] + empty only on table/image.

Does NOT compose a training mix. ~24 pages, stratified none/browser/office.

  export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
  uv run python data/scripts/build_q1_preview.py
  # open data/review_galleries/preview_q1_train/index.html
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

from apply_content_pack import (  # noqa: E402
    OCCUPANCY_BANDS,
    densify_static_html,
    fill_from_pool,
    occupancy_layout,
)
from point_ocr.build_point import load_blocks_json  # noqa: E402
from point_ocr.chrome.wrap import (  # noqa: E402
    BROWSER_SKINS,
    OFFICE_SKINS,
    ChromeSpec,
    sample_chrome_spec,
    wrap_page_html,
)
from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.marker import CURRENT_MARKER_TAG  # noqa: E402
from point_ocr.noise import apply_screen_noise  # noqa: E402
from point_ocr.pools.emit import emit_pool_samples  # noqa: E402
from point_ocr.pools.select import (  # noqa: E402
    OCC_MAX,
    OCC_MIN,
    occupancy_in_train_range,
    text_occupancy_frac,
)
from point_ocr.pools.spec import (  # noqa: E402
    DOC_TEMPLATES,
    SPECIAL_EMPTY_TEMPLATES,
    STATIC_HTML_FILES,
)
from point_ocr.synth.render import HtmlRenderSession  # noqa: E402
from point_ocr.synth.window_viewport import sample_window_capture  # noqa: E402

PROMPT_KEY = "a2_v3"


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


def _page_plan() -> list[dict[str, Any]]:
    docs = list(DOC_TEMPLATES)
    specials = list(SPECIAL_EMPTY_TEMPLATES)
    # Formula/code pages as normal text scenery (not emptied).
    formula_code = ("03_code_ui.html", "07_formulas_defs.html")
    rows: list[dict[str, Any]] = []
    for i in range(6):
        rows.append(
            {
                "kind": "none",
                "skin": None,
                "template": docs[i % len(docs)],
                "band": OCCUPANCY_BANDS[i % len(OCCUPANCY_BANDS)],
            }
        )
    i = 0
    for skin in BROWSER_SKINS:
        rows.append(
            {
                "kind": "browser",
                "skin": skin,
                "template": docs[i % len(docs)],
                "band": OCCUPANCY_BANDS[i % len(OCCUPANCY_BANDS)],
            }
        )
        i += 1
    for skin in OFFICE_SKINS:
        rows.append(
            {
                "kind": "office",
                "skin": skin,
                "template": docs[i % len(docs)],
                "band": OCCUPANCY_BANDS[i % len(OCCUPANCY_BANDS)],
            }
        )
        i += 1
    chrome_cycle = [
        ("none", None),
        ("browser", "chrome_win_light"),
        ("office", "word_win"),
    ]
    for j, tmpl in enumerate(specials):
        kind, skin = chrome_cycle[j % len(chrome_cycle)]
        rows.append(
            {
                "kind": kind,
                "skin": skin,
                "template": tmpl,
                "band": OCCUPANCY_BANDS[j % len(OCCUPANCY_BANDS)],
            }
        )
    for j, tmpl in enumerate(formula_code):
        rows.append(
            {
                "kind": "none",
                "skin": None,
                "template": tmpl,
                "band": OCCUPANCY_BANDS[j % len(OCCUPANCY_BANDS)],
            }
        )
    return rows


def _pick_core(samples: list[PointSample]) -> list[PointSample]:
    dia = [s for s in samples if s.meta.get("region") == "diamond"]
    cor = [s for s in samples if s.meta.get("region") in {"nw", "ne", "sw", "se"}]
    out: list[PointSample] = []
    if dia:
        out.append(dia[0])
    if cor:
        out.append(cor[0])
    if not out:
        out = samples[:2]
    return out[:2]


def _pick_one(samples: list[PointSample], rng: random.Random) -> list[PointSample]:
    if not samples:
        return []
    return [rng.choice(samples)]


def _render_page(
    *,
    session: HtmlRenderSession,
    tmpl: Path,
    page_id: str,
    band: str,
    content_pool: dict[str, Any],
    chrome_spec: ChromeSpec,
    local: random.Random,
    filled_dir: Path,
    render_dir: Path,
    seed: int,
    index: int,
) -> tuple[Image.Image, list, dict[str, Any], str, float, str | None] | None:
    html_src = tmpl.read_text(encoding="utf-8")
    extras = int(occupancy_layout(band)["extra"])
    occ_band = band
    img = None
    meta: dict[str, Any] = {}
    blocks: list = []
    filled = html_src
    layout_name: str | None = None
    occ = 0.0
    for attempt in range(4):
        if tmpl.name in STATIC_HTML_FILES:
            filled, layout_name = densify_static_html(
                html_src,
                content_pool,
                random.Random(seed + index * 17 + attempt * 91),
                occupancy_band=str(occ_band),
                extra_paragraphs=extras,
            )
        else:
            filled, layout_name = fill_from_pool(
                html_src,
                content_pool,
                random.Random(seed + index * 17 + attempt * 91),
                extra_paragraphs=extras,
                layout_set="dense",
                occupancy_band=str(occ_band),
            )
        wrapped = wrap_page_html(filled, chrome_spec)
        html_path = filled_dir / f"{page_id}__a{attempt}.html"
        html_path.write_text(wrapped, encoding="utf-8")
        cap = sample_window_capture(local)
        meta = session.render(
            html_path,
            render_dir,
            page_id=f"{page_id}__a{attempt}",
            viewport=cap.viewport,
            device_scale_factor=cap.device_scale_factor,
            full_page=cap.full_page,
        )
        img = apply_screen_noise(
            Image.open(meta["image_path"]).convert("RGB"),
            rng=random.Random(seed + index * 31 + attempt),
        )
        blocks = load_blocks_json(Path(meta["blocks_path"]))
        occ = text_occupancy_frac(blocks, img.size[0], img.size[1], include_special=True)
        if occupancy_in_train_range(occ):
            return img, blocks, meta, wrapped, occ, layout_name
        if occ < OCC_MIN:
            extras += 8
            occ_band = "occ_60"
        else:
            extras = max(2, extras // 2)
            occ_band = "occ_30"
    return None


def _write_gallery(
    *,
    out: Path,
    pages: list[dict[str, Any]],
    groups: dict[str, list[PointSample]],
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
            · {html.escape(pg['skin'])} · {html.escape(pg['template'])}
            · occ={pg['occ']:.1%}</figcaption>
            <a href="{html.escape(rel)}" target="_blank"><img src="{html.escape(rel)}" /></a>
            </figure>"""
        )

    def _cards(samples: list[PointSample], prefix: str) -> str:
        bits = []
        for i, s in enumerate(samples):
            src = Path(s.image_path)
            dst = img_dir / f"{prefix}_{i:03d}.jpg"
            im = Image.open(src).convert("RGB")
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
            occ = meta.get("text_occupancy")
            occ_s = f"occ={float(occ):.1%}" if occ is not None else "occ=?"
            bits.append(
                f"""<section class="card">
                <div class="meta">
                  <b>#{i}</b> <code>{html.escape(s.sample_id)}</code>
                  <span class="pill {'neg' if neg else 'pos'}">{'负例' if neg else '正例'}</span>
                  <span class="pill">{html.escape(str(meta.get('chrome_skin') or 'none'))}</span>
                  <span class="pill">region={html.escape(str(meta.get('region')))}</span>
                  <span class="pill">{html.escape(str(meta.get('pool_id') or ''))}</span>
                  <span class="pill">{html.escape(occ_s)}</span>
                </div>
                <div class="split">
                  <a href="{html.escape(rel)}" target="_blank"><img src="{html.escape(rel)}" /></a>
                  <pre>{html.escape(gt)}</pre>
                </div>
                </section>"""
            )
        return "".join(bits) or "<p class='meta'>（这一组没有抽到样本）</p>"

    n_samp = sum(len(v) for v in groups.values())
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<title>Q1 train preview</title>
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
<h1>Q1 训练规则预览</h1>
<p>提示词 a2_v3。空串仅：明显空白 / 壳 / 表格 / 图片。公式与代码按正文输出。
页面 ink 并集占用率限制在 {OCC_MIN:.0%}–{OCC_MAX:.0%}。绿框仅审查用。</p>
<p>页 {len(pages)} · 标记样本 {n_samp} · marker={CURRENT_MARKER_TAG} · prompt={PROMPT_KEY}</p>
</header>
<section class="block">
<h2>1. 整页（occ 达标；无十字）</h2>
<div class="grid">{''.join(page_cards)}</div>
</section>
<section class="block">
<h2>2. 正文正点（core_inner；可含公式/代码）</h2>
{_cards(groups['core'], 'core')}
</section>
<section class="block">
<h2>3. 明显空白（empty_clear，离文字 AABB ≥48px）</h2>
{_cards(groups['clear'], 'clear')}
</section>
<section class="block">
<h2>4. 点在表格 / 图片上（empty_special → 空串）</h2>
{_cards(groups['special'], 'special')}
</section>
<section class="block">
<h2>5. 壳带空标（每页最多 1 个）</h2>
{_cards(groups['chrome'], 'chrome')}
</section>
</body></html>"""
    (out / "index.html").write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/preview_q1_train")
    ap.add_argument("--gallery", type=Path, default=ROOT / "data/review_galleries/preview_q1_train")
    ap.add_argument("--content-pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--chrome-pool", type=Path, default=ROOT / "data/synth/content_pools/chrome_pool.json")
    ap.add_argument("--seed", type=int, default=19)
    args = ap.parse_args()
    pw = Path.home() / ".cache" / "ms-playwright"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(pw)

    content_pool = _load_json_pool(args.content_pool)
    chrome_pool = _load_chrome_pool(args.chrome_pool)
    plan = _page_plan()
    rng = random.Random(args.seed)

    filled_dir = args.out / "filled_html"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"
    for d in (filled_dir, render_dir, marked_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    pages_meta: list[dict[str, Any]] = []
    groups: dict[str, list[PointSample]] = {
        "core": [],
        "clear": [],
        "special": [],
        "chrome": [],
    }
    all_samples: list[PointSample] = []
    skipped = 0

    tmpl_root = ROOT / "data/synth/templates"
    with HtmlRenderSession(wait_until="load") as session:
        for i, row in enumerate(plan):
            tmpl = tmpl_root / row["template"]
            page_id = f"q1t_{i:02d}_{row['kind']}_{row['skin'] or 'none'}_{tmpl.stem}"
            local = random.Random(args.seed + i * 97)
            spec = _forced_spec(row["kind"], row["skin"], local, chrome_pool)
            rendered = _render_page(
                session=session,
                tmpl=tmpl,
                page_id=page_id,
                band=str(row["band"]),
                content_pool=content_pool,
                chrome_spec=spec,
                local=local,
                filled_dir=filled_dir,
                render_dir=render_dir,
                seed=args.seed,
                index=i,
            )
            if rendered is None:
                skipped += 1
                print(f"[{i+1}/{len(plan)}] SKIP {page_id} (occ out of range)", flush=True)
                continue
            img, blocks, meta, wrapped, occ, layout = rendered
            chrome_path = Path(meta.get("chrome_path") or "")
            bands: list[dict[str, Any]] = []
            if chrome_path.is_file():
                bands = json.loads(chrome_path.read_text(encoding="utf-8"))
            capture_meta = {
                **spec.to_meta(),
                "text_occupancy": round(occ, 4),
                "occupancy_band": row["band"],
            }
            emit_kw = dict(
                img=img,
                blocks=blocks,
                page_id=page_id,
                marked_dir=marked_dir,
                seed=args.seed + i * 7,
                template_stem=tmpl.stem,
                prompt_key=PROMPT_KEY,
                layout_name=layout,
                capture_meta=capture_meta,
                page_html=wrapped,
                chrome_bands=bands,
            )
            core = emit_pool_samples(**emit_kw, pool_id="core_inner")
            clear = emit_pool_samples(**emit_kw, pool_id="empty_clear")
            special = emit_pool_samples(**emit_kw, pool_id="empty_special")
            chrome_negs = [
                s for s in clear if str(s.meta.get("region") or "").startswith("chrome_")
            ]
            far = [
                s for s in clear if not str(s.meta.get("region") or "").startswith("chrome_")
            ]
            picked_core = _pick_core(core)
            picked_clear = _pick_one(far, local)
            picked_special = special[:2]
            picked_chrome = chrome_negs[:1]
            groups["core"].extend(picked_core)
            groups["clear"].extend(picked_clear)
            groups["special"].extend(picked_special)
            groups["chrome"].extend(picked_chrome)
            all_samples.extend(picked_core + picked_clear + picked_special + picked_chrome)
            pages_meta.append(
                {
                    "page_id": page_id,
                    "render": meta["image_path"],
                    "skin": spec.skin,
                    "template": tmpl.name,
                    "occ": occ,
                }
            )
            print(
                f"[{i+1}/{len(plan)}] {page_id} skin={spec.skin} occ={occ:.3f} "
                f"core={len(core)} clear={len(far)} special={len(special)} chrome={len(chrome_negs)}",
                flush=True,
            )

    jsonl = args.out / "point_sharegpt.jsonl"
    write_jsonl(jsonl, all_samples)
    _write_gallery(out=args.gallery, pages=pages_meta, groups=groups)
    print(f"jsonl {jsonl} n={len(all_samples)} skipped_pages={skipped}", flush=True)
    print(f"gallery {args.gallery / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

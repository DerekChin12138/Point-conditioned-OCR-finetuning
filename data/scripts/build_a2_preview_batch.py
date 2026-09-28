#!/usr/bin/env python3
"""Build a stratified A2 preview batch covering every new curriculum slice.

Uses A2 scatter-star marker + A2 POINT_PROMPT. Writes jsonl + review gallery.

Example:
  uv run python data/scripts/build_a2_preview_batch.py \\
    --out data/processed/preview_a2 \\
    --gallery data/review_galleries/preview_a2
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from PIL import Image  # noqa: E402

from apply_content_pack import fill_from_pool  # noqa: E402
from point_ocr.a2_labels import (  # noqa: E402
    build_semantic_group_map,
    classify_a2_kind,
    is_block_visibly_clear,
    is_focused_window_block,
    parse_semantic_groups,
    resolve_a2_target,
    rewrite_block_for_a2,
    select_desktop_positive_blocks,
)
from point_ocr.a2_mix import (  # noqa: E402
    A2_SLICE_FRAC,
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
from point_ocr.sample_points import BBox, sample_points_in_block  # noqa: E402
from point_ocr.synth.render import render_html_file_sync  # noqa: E402

REPLAY_TEMPLATES = [
    "01_article_twocol.html",
    "08_github_readme_dark.html",
    "12_wikipedia_article.html",
    "20_forum_zh.html",
    "24_docs_portal_dense.html",
]
ADJ_TEMPLATES = [
    "01_article_twocol.html",
    "04_zh_news_twocol.html",
    "14_magazine_3col.html",
]
MULTI_TEMPLATES = [
    "01_article_twocol.html",
    "14_magazine_3col.html",
    "04_zh_news_twocol.html",
]
DESKTOP_TEMPLATES = [
    "21_desktop_stacked_windows.html",
    "22_desktop_zh_messy.html",
    "25_desktop_win11_collage.html",
    "26_desktop_mac_collage.html",
    "23_ide_dense_split.html",
]


def _load_pool(path: Path) -> dict:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pool") or data


def _fill(
    template: Path,
    pool: dict,
    out_html: Path,
    *,
    seed: int,
    extras: int,
    layout_set: str,
) -> str:
    html_src = template.read_text(encoding="utf-8")
    filled, layout = fill_from_pool(
        html_src,
        pool,
        random.Random(seed),
        extra_paragraphs=extras,
        layout_set=layout_set,
    )
    out_html.write_text(filled, encoding="utf-8")
    return layout


def _render_page(
    html_path: Path,
    render_dir: Path,
    page_id: str,
    *,
    use_noise: bool,
    seed: int,
) -> tuple[Image.Image, list[BlockAnno], Path, str]:
    meta = render_html_file_sync(html_path, render_dir, page_id=page_id)
    img = Image.open(meta["image_path"]).convert("RGB")
    if use_noise:
        img = apply_screen_noise(img, rng=random.Random(seed))
    blocks = load_blocks_json(Path(meta["blocks_path"]))
    page_html = html_path.read_text(encoding="utf-8")
    group_map = build_semantic_group_map(page_html)
    # A2: table→HTML; semantic-group→aggregate; formula/code/image→empty
    blocks = [rewrite_block_for_a2(b, page_html, group_map) for b in blocks]
    return img, blocks, Path(meta["image_path"]), page_html


def _large_blocks(
    blocks: list[BlockAnno],
    page_area: float,
    *,
    min_area: float = 0.015,
    min_chars: int = 40,
    prefer_multi: bool = False,
) -> list[BlockAnno]:
    kept: list[BlockAnno] = []
    for b in blocks:
        fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
        if not fr.keep:
            continue
        if b.ink_area() / page_area < min_area:
            continue
        if len(b.markdown.strip()) < min_chars:
            continue
        kept.append(b)
    if prefer_multi:
        multi = [b for b in kept if len(b.ink_rects()) > 1]
        if multi:
            kept = multi
    kept.sort(key=lambda b: b.ink_area(), reverse=True)
    return kept


def _tag_slice(samples: list[PointSample], slice_name: str) -> list[PointSample]:
    for s in samples:
        s.meta["a2_slice"] = slice_name
        s.meta["marker"] = "a2"
        if not s.meta.get("template_stem"):
            s.meta["template_stem"] = template_stem_from_page_id(str(s.meta.get("page_id") or ""))
    return samples


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
) -> PointSample | None:
    w, h = img.size
    pts = sample_points_in_block(
        block.ink_rects()[0] if block.ink_rects() else block.bbox,
        n=1,
        coverage=coverage,
        rng=random.Random(seed + idx),
        block_id=block.block_id,
        image_w=w,
        image_h=h,
        fragment_index=0,
    )
    # Prefer multi-frag: sample via fragments for multi
    if len(block.ink_rects()) > 1:
        from point_ocr.sample_points import sample_points_in_fragments

        pts = sample_points_in_fragments(
            block.ink_rects(),
            n=1,
            coverage=coverage,
            rng=random.Random(seed + idx),
            block_id=block.block_id,
            image_w=w,
            image_h=h,
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
    return PointSample(
        sample_id=f"{page_id}:{slice_name}:{block.block_id}:p{idx}",
        image_path=str(path.resolve()),
        task="POINT",
        target=block.markdown,
        meta={
            "page_id": page_id,
            "block_id": block.block_id,
            "point": [pt.x, pt.y],
            "point_norm": [nx, ny],
            "image_w": w,
            "image_h": h,
            "region": pt.region,
            "bbox": list(frag.as_tuple()),
            "bboxes": [list(r.as_tuple()) for r in block.ink_rects()],
            "bbox_union": list(block.bbox.as_tuple()),
            "n_fragments": len(block.ink_rects()),
            "fragment_index": pt.fragment_index,
            # Deny (formula/code/image) keeps empty target but is NOT a chrome negative
            "is_negative": False,
            "a2_slice": slice_name,
            "marker": "a2",
            "template_stem": template_stem_from_page_id(page_id),
            **{k: v for k, v in (block.extra or {}).items() if k in {"a2_kind", "group_id", "group_role", "group_kind"}},
        },
    )


def _neg_near_edge(
    img: Image.Image,
    avoid: list[BBox],
    *,
    page_id: str,
    out_dir: Path,
    seed: int,
    n: int = 2,
    band: float = 36.0,
) -> list[PointSample]:
    """Negatives inside the near-border band (chrome / margin stress)."""
    rng = random.Random(seed)
    w, h = img.size
    out: list[PointSample] = []
    tries = 0
    while len(out) < n and tries < 600:
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
                },
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--templates-dir", type=Path, default=ROOT / "data/synth/templates")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/preview_a2")
    ap.add_argument("--gallery", type=Path, default=ROOT / "data/review_galleries/preview_a2")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--skip-gallery", action="store_true")
    args = ap.parse_args()
    use_noise = not args.no_noise

    if not args.pool.is_file():
        raise SystemExit(f"Missing pool {args.pool}")

    pool = _load_pool(args.pool)
    rng = random.Random(args.seed)
    filled_dir = args.out / "filled_html"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"
    for d in (filled_dir, render_dir, marked_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_samples: list[PointSample] = []
    slice_counts: Counter[str] = Counter()

    def add(samples: list[PointSample], slice_name: str) -> None:
        _tag_slice(samples, slice_name)
        all_samples.extend(samples)
        slice_counts[slice_name] += len(samples)

    # ---- 1) A1 replay (center_band, large blocks) ----
    for i, name in enumerate(REPLAY_TEMPLATES[:4]):
        tmpl = args.templates_dir / name
        if not tmpl.is_file():
            continue
        page_id = f"replay__{tmpl.stem}__{i}"
        filled = filled_dir / f"{page_id}.html"
        _fill(tmpl, pool, filled, seed=args.seed + i * 11, extras=5, layout_set="dense")
        img, blocks, _, _html = _render_page(filled, render_dir, page_id, use_noise=use_noise, seed=args.seed + i)
        w, h = img.size
        kept = _large_blocks(blocks, float(w * h))[:3]
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=1,
            seed=args.seed + i,
            avoid_boxes=[r for b in blocks for r in b.ink_rects()],
            coverage="center_band",
            marker="a2",
            meta_extra={"a2_slice": "replay"},
        )
        pos = [s for s in batch if not s.meta.get("is_negative")][:3]
        neg = [s for s in batch if s.meta.get("is_negative")][:1]
        add(pos + neg, "replay")
        print(f"[replay] {page_id}: +{len(pos)} -{len(neg)}")

    # ---- 2) Adjacency / edge_band ----
    for i, name in enumerate(ADJ_TEMPLATES):
        tmpl = args.templates_dir / name
        if not tmpl.is_file():
            continue
        page_id = f"adj__{tmpl.stem}__{i}"
        filled = filled_dir / f"{page_id}.html"
        _fill(tmpl, pool, filled, seed=args.seed + 100 + i * 13, extras=6, layout_set="dense")
        img, blocks, _, _html = _render_page(filled, render_dir, page_id, use_noise=use_noise, seed=args.seed + 100 + i)
        w, h = img.size
        kept = _large_blocks(blocks, float(w * h), min_area=0.01)[:4]
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=1,
            seed=args.seed + 200 + i,
            avoid_boxes=[r for b in blocks for r in b.ink_rects()],
            coverage="edge_band",
            marker="a2",
            meta_extra={"a2_slice": "adjacency"},
        )
        pos = [s for s in batch if not s.meta.get("is_negative")][:4]
        neg = [s for s in batch if s.meta.get("is_negative")][:1]
        add(pos + neg, "adjacency")
        print(f"[adjacency] {page_id}: +{len(pos)} -{len(neg)}")

    # ---- 3) Multi-fragment ----
    for i, name in enumerate(MULTI_TEMPLATES + ["14_magazine_3col.html", "01_article_twocol.html"]):
        tmpl = args.templates_dir / name
        if not tmpl.is_file():
            continue
        page_id = f"mfrag__{tmpl.stem}__{i}"
        filled = filled_dir / f"{page_id}.html"
        _fill(tmpl, pool, filled, seed=args.seed + 300 + i * 17, extras=8, layout_set="dense")
        img, blocks, _, _html = _render_page(filled, render_dir, page_id, use_noise=use_noise, seed=args.seed + 300 + i)
        w, h = img.size
        # Prefer true multi-rect ink; relax area so column-split paras survive
        multi = [b for b in blocks if len(b.ink_rects()) > 1 and len(b.markdown.strip()) >= 30]
        multi.sort(key=lambda b: b.ink_area(), reverse=True)
        kept = multi[:4] or _large_blocks(blocks, float(w * h), prefer_multi=True)[:2]
        n_got = 0
        for j, b in enumerate(kept):
            s = _emit_one(
                img,
                b,
                page_id=page_id,
                out_dir=marked_dir,
                coverage="center_band",
                seed=args.seed + 310 + i,
                slice_name="multi_frag",
                idx=j,
            )
            if s:
                s.meta["n_fragments"] = len(b.ink_rects())
                all_samples.append(s)
                slice_counts["multi_frag"] += 1
                n_got += 1
        print(f"[multi_frag] {page_id}: +{n_got} (multi_blocks={len(multi)})")

    # ---- 4) Semantic groups (bidirectional) ----
    sg_tmpl = args.templates_dir / "27_a2_semantic_groups.html"
    if sg_tmpl.is_file():
        page_id = "semgroup__27"
        filled = filled_dir / f"{page_id}.html"
        # Static template — no pool fill needed, but layout inject is fine
        filled.write_text(sg_tmpl.read_text(encoding="utf-8"), encoding="utf-8")
        img, blocks, _, _html = _render_page(filled, render_dir, page_id, use_noise=False, seed=args.seed + 400)
        by_id = {b.block_id: b for b in blocks}
        groups = parse_semantic_groups(filled.read_text(encoding="utf-8"))
        n_seed = n_body = 0
        for gi, g in enumerate(groups):
            # 40% seed / 60% body — force both kinds across groups
            pick_body = (gi % 5) != 0  # 4/5 body → ~80% body for preview stress; mix with seeds
            if gi % 3 == 0:
                pick_body = False
            cand_ids = g.body_ids if pick_body and g.body_ids else g.seed_ids
            if not cand_ids:
                cand_ids = g.member_ids
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
                seed=args.seed + 410 + gi,
                slice_name="semantic_group",
                idx=gi,
            )
            if s:
                all_samples.append(s)
                slice_counts["semantic_group"] += 1
                if pick_body:
                    n_body += 1
                else:
                    n_seed += 1
        # orphan short (no expand)
        orphan = by_id.get("orphan_nav")
        if orphan:
            s = _emit_one(
                img,
                orphan,
                page_id=page_id,
                out_dir=marked_dir,
                coverage="center_band",
                seed=args.seed + 499,
                slice_name="semantic_group",
                idx=99,
            )
            if s:
                s.meta["group_role"] = "orphan"
                all_samples.append(s)
                slice_counts["semantic_group"] += 1
        print(f"[semantic_group] groups={len(groups)} seed={n_seed} body={n_body}")

    # ---- 5) Corner extreme ----
    for i, name in enumerate(REPLAY_TEMPLATES[:3]):
        tmpl = args.templates_dir / name
        if not tmpl.is_file():
            continue
        page_id = f"corner__{tmpl.stem}__{i}"
        filled = filled_dir / f"{page_id}.html"
        _fill(tmpl, pool, filled, seed=args.seed + 500 + i * 19, extras=4, layout_set="dense")
        img, blocks, _, _html = _render_page(filled, render_dir, page_id, use_noise=use_noise, seed=args.seed + 500 + i)
        w, h = img.size
        kept = _large_blocks(blocks, float(w * h))[:3]
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=0,
            seed=args.seed + 520 + i,
            avoid_boxes=[r for b in blocks for r in b.ink_rects()],
            coverage="corner_extreme",
            marker="a2",
            meta_extra={"a2_slice": "corner_extreme"},
        )
        pos = [s for s in batch if not s.meta.get("is_negative")][:3]
        add(pos, "corner_extreme")
        print(f"[corner_extreme] {page_id}: +{len(pos)}")

    # ---- 6) Special semantics (table / deny / prose control) ----
    sp_tmpl = args.templates_dir / "28_a2_special_semantics.html"
    if sp_tmpl.is_file():
        page_id = "special__28"
        filled = filled_dir / f"{page_id}.html"
        filled.write_text(sp_tmpl.read_text(encoding="utf-8"), encoding="utf-8")
        img, blocks, _, _html = _render_page(filled, render_dir, page_id, use_noise=False, seed=args.seed + 600)
        for j, b in enumerate(blocks):
            kind = str((b.extra or {}).get("a2_kind") or "text")
            # Skip pure section headings
            if kind == "text" and b.block_id in {
                "sp_title",
                "sp_meta",
                "sp_h_table",
                "sp_h_formula",
                "sp_h_code",
                "sp_h_img",
            }:
                continue
            slice_name = {
                "table": "special_table",
                "formula": "special_deny",
                "code": "special_deny",
                "image": "special_deny",
            }.get(kind, "special_prose")
            s = _emit_one(
                img,
                b,
                page_id=page_id,
                out_dir=marked_dir,
                coverage="center_band",
                seed=args.seed + 610 + j,
                slice_name=slice_name,
                idx=j,
            )
            if s:
                all_samples.append(s)
                slice_counts[slice_name] += 1
        print(f"[special] table/deny/prose from {page_id}")

    # Also formula template for extra deny coverage
    for i, name in enumerate(["07_formulas_defs.html", "03_code_ui.html"]):
        tmpl = args.templates_dir / name
        if not tmpl.is_file():
            continue
        page_id = f"deny__{tmpl.stem}"
        filled = filled_dir / f"{page_id}.html"
        filled.write_text(tmpl.read_text(encoding="utf-8"), encoding="utf-8")
        img, blocks, _, _html = _render_page(filled, render_dir, page_id, use_noise=False, seed=args.seed + 650 + i)
        n_d = 0
        for j, b in enumerate(blocks):
            kind = str((b.extra or {}).get("a2_kind") or "")
            if b.extra and b.extra.get("data-latex"):
                kind = "formula"
            if kind not in {"formula", "code"}:
                continue
            # Ensure empty GT for deny
            labeled = BlockAnno(
                block_id=b.block_id,
                bbox=b.bbox,
                markdown="",
                rects=b.rects,
                extra={**(b.extra or {}), "a2_kind": kind},
            )
            s = _emit_one(
                img,
                labeled,
                page_id=page_id,
                out_dir=marked_dir,
                coverage="center_band",
                seed=args.seed + 660 + i * 10 + j,
                slice_name="special_deny",
                idx=j,
            )
            if s:
                all_samples.append(s)
                slice_counts["special_deny"] += 1
                n_d += 1
        print(f"[special_deny] {page_id}: {n_d}")

    # ---- 7) Desktop shell ----
    for i, name in enumerate(DESKTOP_TEMPLATES):
        tmpl = args.templates_dir / name
        if not tmpl.is_file():
            continue
        page_id = f"desk__{tmpl.stem}"
        filled = filled_dir / f"{page_id}.html"
        # Do not pool-fill desktop shells — preserves coherent tables/windows.
        filled.write_text(tmpl.read_text(encoding="utf-8"), encoding="utf-8")
        img, blocks, _, _html = _render_page(filled, render_dir, page_id, use_noise=False, seed=args.seed + 700 + i)
        # Desktop: keep short UI strings + full-table GT; occlusion/focus is the gate.
        candidates: list[BlockAnno] = []
        for b in blocks:
            md = (b.markdown or "").strip()
            kind = str((b.extra or {}).get("a2_kind") or "")
            if kind in {"formula", "code", "image"}:
                continue  # deny kinds not used as desktop positives here
            if len(md) < 2:
                continue
            candidates.append(b)
        # Prefer semantic-group short+context when available (web-shell short blocks).
        group_cands = [
            b
            for b in candidates
            if (b.extra or {}).get("a2_kind") == "semantic_group"
            and is_block_visibly_clear(b.extra, min_visible_frac=0.9)
        ]
        focus_groups = [b for b in group_cands if is_focused_window_block(b.extra)]
        pool_for_select = focus_groups or group_cands or candidates
        kept = select_desktop_positive_blocks(
            pool_for_select,
            max_n=3,
            min_visible_frac=0.9,
            focused_frac=0.9,
        )
        kept = [
            b
            for b in kept
            if (b.extra or {}).get("visible_frac") is None
            or float(b.extra.get("visible_frac") or 0) >= 0.9
        ][:3]
        if not kept:
            print(f"[desktop] {page_id}: no clear/focused blocks; skip positives")
            batch = build_point_samples_for_page(
                img,
                [],
                page_id=page_id,
                out_image_dir=marked_dir,
                r_min=1,
                r_max=1,
                n_negatives=2,
                seed=args.seed + 720 + i,
                avoid_boxes=[r for b in blocks for r in b.ink_rects()],
                coverage="center_band",
                marker="a2",
                meta_extra={"a2_slice": "desktop"},
            )
            neg = [s for s in batch if s.meta.get("is_negative")][:2]
            add(neg, "desktop")
            continue
        batch = build_point_samples_for_page(
            img,
            kept,
            page_id=page_id,
            out_image_dir=marked_dir,
            r_min=1,
            r_max=1,
            n_negatives=2,
            seed=args.seed + 720 + i,
            avoid_boxes=[r for b in blocks for r in b.ink_rects()],
            coverage="center_band",
            marker="a2",
            meta_extra={"a2_slice": "desktop"},
        )
        pos = [s for s in batch if not s.meta.get("is_negative")][:3]
        for s in pos:
            bid = s.meta.get("block_id")
            src = next((b for b in kept if b.block_id == bid), None)
            if src and src.extra:
                s.meta["visible_frac"] = src.extra.get("visible_frac")
                s.meta["window_id"] = src.extra.get("window_id")
                s.meta["window_focused"] = src.extra.get("window_focused")
        neg = [s for s in batch if s.meta.get("is_negative")][:2]
        n_f = sum(1 for s in pos if s.meta.get("window_focused"))
        add(pos + neg, "desktop")
        print(
            f"[desktop] {page_id}: +{len(pos)} -{len(neg)} "
            f"focused={n_f}/{len(pos)} cands={len(candidates)}"
        )

    # ---- 8) Near-edge / chrome negatives ----
    # Reuse a dense page for near-edge negs
    replay_imgs = sorted(render_dir.glob("replay__*.png"))
    if replay_imgs:
        page_id = "near_edge__from_replay"
        img = Image.open(replay_imgs[0]).convert("RGB")
        # Load matching blocks if present
        bp = Path(str(replay_imgs[0]).replace(".png", ".blocks.json"))
        avoid: list[BBox] = []
        if bp.is_file():
            avoid = [r for b in load_blocks_json(bp) for r in b.ink_rects()]
        negs = _neg_near_edge(img, avoid, page_id=page_id, out_dir=marked_dir, seed=args.seed + 800, n=6)
        all_samples.extend(negs)
        slice_counts["near_edge"] += len(negs)
        print(f"[near_edge] {len(negs)} negatives")

    # Scatter by (slice, template) so train/inspect order is interleaved, not
    # clustered by the generation order of each curriculum slice.
    all_samples = interleave_a2_samples(
        all_samples,
        rng=rng,
        slice_fn=lambda s: str(s.meta.get("a2_slice") or "?"),
        template_fn=lambda s: meta_template_stem(s.meta),
    )
    jsonl = args.out / "point_sharegpt.jsonl"
    n = write_jsonl(jsonl, all_samples)
    print(f"\nWrote {n} → {jsonl}")
    print("Slice counts:", dict(sorted(slice_counts.items())))
    print("A2 full-build quotas (n=%d):" % A2_TARGET_DEFAULT, quota_counts(A2_TARGET_DEFAULT))
    print("A2 slice fracs:", dict(A2_SLICE_FRAC))
    mix_stats = summarize_mix([s.to_sharegpt() for s in all_samples])
    print(
        "Mix: max consecutive same (slice,template) =",
        mix_stats["max_consecutive_same_slice_template"],
    )

    if not args.skip_gallery and n > 0:
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "data/scripts/review_sample_gallery.py"),
                "--jsonl",
                str(jsonl),
                "--out",
                str(args.gallery),
                "--n",
                str(min(n, 120)),
                "--seed",
                str(args.seed),
                "--show-bbox",
                "--stratify",
                "a2_slice",
            ]
        )
        print(f"Open gallery: {args.gallery / 'index.html'}")


if __name__ == "__main__":
    main()

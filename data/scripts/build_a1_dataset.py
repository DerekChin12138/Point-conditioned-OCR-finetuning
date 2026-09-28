#!/usr/bin/env python3
"""Build Stage-A1 POINT dataset (doc-only, large blocks, center-band points).

Fills templates from the content pool, renders (multi-process Chromium),
samples center-band points until --target samples. Sparse pages are a minority.
Cross-column CSS templates are kept; multi-fragment bboxes sample on ink only.

Example:
  uv run python data/scripts/build_a1_dataset.py --target 15000 --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from PIL import Image  # noqa: E402

from apply_content_pack import fill_from_pool  # noqa: E402
from point_ocr.build_point import BlockAnno, build_point_samples_for_page, load_blocks_json  # noqa: E402
from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.filter_qa import filter_block_label  # noqa: E402
from point_ocr.noise import apply_screen_noise  # noqa: E402
from point_ocr.synth.render import HtmlRenderSession  # noqa: E402

DENSE_TEMPLATES = [
    "01_article_twocol.html",
    "04_zh_news_twocol.html",
    "08_github_readme_dark.html",
    "12_wikipedia_article.html",
    "14_magazine_3col.html",
    "20_forum_zh.html",
    "24_docs_portal_dense.html",
]
SPARSE_TEMPLATES = [
    "05_pdf_academic.html",
    "15_slides_dark.html",
    "11_email_client.html",
]

# Per-process Chromium session (Playwright is not shareable across processes).
_WORKER: dict[str, Any] = {}


def _worker_init(pool_path: str) -> None:
    data = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    pool = data.get("pool") or data
    _WORKER["pool"] = pool
    _WORKER["session"] = HtmlRenderSession(wait_until="load").__enter__()


def _worker_close() -> None:
    session = _WORKER.pop("session", None)
    if session is not None:
        session.__exit__(None, None, None)


def _process_page(job: dict[str, Any]) -> dict[str, Any]:
    """Fill → render → sample one page. Returns serializable sample dicts."""
    pool = _WORKER["pool"]
    session: HtmlRenderSession = _WORKER["session"]

    page_id = job["page_id"]
    tmpl_path = Path(job["template"])
    filled_dir = Path(job["filled_dir"])
    render_dir = Path(job["render_dir"])
    marked_dir = Path(job["marked_dir"])
    use_sparse = bool(job["use_sparse"])
    extras = int(job["extras"])
    layout_set = job["layout_set"]
    seed = int(job["seed"])
    i = int(job["index"])
    use_noise = bool(job["use_noise"])
    min_area_ratio = float(job["min_area_ratio"])
    min_chars = int(job["min_chars"])
    points_per_page = int(job["points_per_page"])
    n_negatives = int(job["n_negatives"])

    html_src = tmpl_path.read_text(encoding="utf-8")
    filled_html, layout_name = fill_from_pool(
        html_src,
        pool,
        random.Random(seed + i * 17),
        extra_paragraphs=extras,
        layout_set=layout_set,
    )
    filled_path = filled_dir / f"{page_id}.html"
    filled_path.write_text(filled_html, encoding="utf-8")

    meta = session.render(filled_path, render_dir, page_id=page_id)
    img = Image.open(meta["image_path"]).convert("RGB")
    if use_noise:
        img = apply_screen_noise(img, rng=random.Random(seed + i * 31))
    w, h = img.size
    page_area = float(w * h)

    min_area = min_area_ratio * (0.6 if use_sparse else 1.0)
    min_chars_eff = max(20, min_chars - (20 if use_sparse else 0))

    blocks = load_blocks_json(Path(meta["blocks_path"]))
    all_boxes = [r for b in blocks for r in b.ink_rects()]
    kept: list[BlockAnno] = []
    for b in blocks:
        fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
        if not fr.keep:
            continue
        if b.ink_area() / page_area < min_area:
            continue
        if len(b.markdown.strip()) < min_chars_eff:
            continue
        kept.append(b)
    kept.sort(key=lambda b: b.ink_area(), reverse=True)
    kept = kept[: max(3, points_per_page)]

    if not kept:
        return {
            "page_id": page_id,
            "skipped": True,
            "use_sparse": use_sparse,
            "kept": 0,
            "n_multi": 0,
            "pos": [],
            "neg": [],
        }

    batch = build_point_samples_for_page(
        img,
        kept,
        page_id=page_id,
        out_image_dir=marked_dir,
        r_min=1,
        r_max=2,
        n_negatives=n_negatives,
        seed=seed + i,
        avoid_boxes=all_boxes,
        coverage="center_band",
        band_half_frac=0.35,
    )
    density = "sparse" if use_sparse else "dense"
    pos: list[dict[str, Any]] = []
    neg: list[dict[str, Any]] = []
    rng = random.Random(seed + i * 7)
    pos_s = [s for s in batch if not s.meta.get("is_negative")]
    neg_s = [s for s in batch if s.meta.get("is_negative")]
    rng.shuffle(pos_s)
    pos_s = pos_s[:points_per_page]
    for s in pos_s + neg_s:
        s.meta["density"] = density
        s.meta["layout"] = layout_name
        row = asdict(s)
        if s.meta.get("is_negative"):
            neg.append(row)
        else:
            pos.append(row)

    n_multi = sum(1 for b in kept if len(b.ink_rects()) > 1)
    return {
        "page_id": page_id,
        "skipped": False,
        "use_sparse": use_sparse,
        "kept": len(kept),
        "n_multi": n_multi,
        "pos": pos,
        "neg": neg,
    }


def _dict_to_sample(d: dict[str, Any]) -> PointSample:
    return PointSample(
        sample_id=d["sample_id"],
        image_path=d["image_path"],
        task=d["task"],
        target=d["target"],
        meta=d.get("meta") or {},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--templates-dir", type=Path, default=ROOT / "data/synth/templates")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/synth")
    ap.add_argument("--target", type=int, default=15000)
    ap.add_argument("--points-per-page", type=int, default=5)
    ap.add_argument("--negatives", type=int, default=1, help="Negatives per page that has ≥1 positive")
    ap.add_argument(
        "--neg-frac",
        type=float,
        default=0.15,
        help="Final negative share of --target (downsample if over)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area-ratio", type=float, default=0.02)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--extra-paragraphs", type=int, default=6)
    ap.add_argument("--sparse-extra-paragraphs", type=int, default=1)
    ap.add_argument("--sparse-page-frac", type=float, default=0.18)
    ap.add_argument("--noise", action="store_true", default=True)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--max-pages", type=int, default=6000)
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, (os.cpu_count() or 4) // 2)),
        help="Parallel Chromium worker processes (each owns a browser)",
    )
    ap.add_argument(
        "--wave-size",
        type=int,
        default=0,
        help="Pages submitted per wave (default: workers * 8)",
    )
    ap.add_argument(
        "--wipe",
        action="store_true",
        help="Delete previous filled_html/renders/marked/jsonl under --out before writing",
    )
    args = ap.parse_args()
    use_noise = args.noise and not args.no_noise
    n_workers = max(1, args.workers)
    wave_size = args.wave_size or (n_workers * 8)

    data = json.loads(args.pool.read_text(encoding="utf-8"))
    pool = data.get("pool") or data
    if not isinstance(pool, dict) or not pool.get("paragraphs"):
        raise SystemExit(f"bad pool: {args.pool}")

    rng = random.Random(args.seed)
    filled_dir = args.out / "filled_html"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"
    if args.wipe:
        for d in (filled_dir, render_dir, marked_dir):
            if d.exists():
                shutil.rmtree(d)
        jsonl_old = args.out / "point_sharegpt.jsonl"
        if jsonl_old.exists():
            jsonl_old.unlink()
        print("[wipe] cleared previous synth outputs under", args.out, flush=True)
    for d in (filled_dir, render_dir, marked_dir):
        d.mkdir(parents=True, exist_ok=True)

    dense = [args.templates_dir / n for n in DENSE_TEMPLATES if (args.templates_dir / n).is_file()]
    sparse = [args.templates_dir / n for n in SPARSE_TEMPLATES if (args.templates_dir / n).is_file()]
    if not dense:
        raise SystemExit("No dense templates")

    pos_samples: list[PointSample] = []
    neg_samples: list[PointSample] = []
    n_sparse_pages = 0
    n_dense_pages = 0
    n_skipped_empty = 0
    n_multi_kept = 0
    n_neg_target = int(round(args.target * args.neg_frac))
    n_pos_target = args.target - n_neg_target

    t0 = time.time()
    print(
        f"[start] target={args.target} pos={n_pos_target} neg={n_neg_target} "
        f"workers={n_workers} wave={wave_size}",
        flush=True,
    )

    next_i = 0
    pages_done = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(args.pool.resolve()),),
    ) as ex:
        pending: dict[Any, int] = {}

        def _enough() -> bool:
            return len(pos_samples) >= n_pos_target and len(neg_samples) >= n_neg_target

        def _submit_wave() -> None:
            nonlocal next_i, n_dense_pages, n_sparse_pages
            if _enough() or next_i >= args.max_pages:
                return
            budget = min(wave_size, args.max_pages - next_i)
            # Overshoot a bit so we don't stall waiting for pos/neg balance
            for _ in range(budget):
                if next_i >= args.max_pages:
                    break
                i = next_i
                next_i += 1
                use_sparse = bool(sparse) and (rng.random() < args.sparse_page_frac)
                if use_sparse:
                    tmpl = rng.choice(sparse)
                    extras = args.sparse_extra_paragraphs
                    layout_set = "sparse"
                    n_sparse_pages += 1
                else:
                    tmpl = dense[i % len(dense)]
                    extras = args.extra_paragraphs
                    layout_set = "dense"
                    n_dense_pages += 1
                page_id = f"{tmpl.stem}__{'sp' if use_sparse else 'dn'}{i}"
                need_neg = len(neg_samples) < n_neg_target
                job = {
                    "page_id": page_id,
                    "template": str(tmpl),
                    "filled_dir": str(filled_dir),
                    "render_dir": str(render_dir),
                    "marked_dir": str(marked_dir),
                    "use_sparse": use_sparse,
                    "extras": extras,
                    "layout_set": layout_set,
                    "seed": args.seed,
                    "index": i,
                    "use_noise": use_noise,
                    "min_area_ratio": args.min_area_ratio,
                    "min_chars": args.min_chars,
                    "points_per_page": args.points_per_page,
                    "n_negatives": args.negatives if need_neg else 0,
                }
                fut = ex.submit(_process_page, job)
                pending[fut] = i

        _submit_wave()
        while pending and not _enough():
            for fut in as_completed(list(pending.keys()), timeout=None):
                pending.pop(fut, None)
                pages_done += 1
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"[warn] page failed: {e}", flush=True)
                    continue

                if result.get("skipped"):
                    n_skipped_empty += 1
                else:
                    n_multi_kept += int(result.get("n_multi") or 0)
                    if len(pos_samples) < n_pos_target:
                        room = n_pos_target - len(pos_samples)
                        for row in result.get("pos") or []:
                            if room <= 0:
                                break
                            pos_samples.append(_dict_to_sample(row))
                            room -= 1
                    if len(neg_samples) < n_neg_target:
                        room = n_neg_target - len(neg_samples)
                        for row in result.get("neg") or []:
                            if room <= 0:
                                break
                            neg_samples.append(_dict_to_sample(row))
                            room -= 1

                if pages_done % 20 == 0 or _enough():
                    elapsed = time.time() - t0
                    rate = pages_done / elapsed if elapsed > 0 else 0
                    print(
                        f"[ok] pages={pages_done} pos={len(pos_samples)}/{n_pos_target} "
                        f"neg={len(neg_samples)}/{n_neg_target} "
                        f"multi_blocks={n_multi_kept} "
                        f"{rate:.1f} pages/s last={result.get('page_id')} "
                        f"kept={result.get('kept')}",
                        flush=True,
                    )

                if _enough():
                    break

            if not _enough() and next_i < args.max_pages:
                _submit_wave()
            elif not pending:
                break

        # Cancel leftovers so workers can exit promptly
        for fut in list(pending.keys()):
            fut.cancel()

    if len(pos_samples) < n_pos_target:
        raise SystemExit(
            f"not enough positives: got {len(pos_samples)} need {n_pos_target}; "
            f"raise --max-pages (skipped_empty={n_skipped_empty})"
        )
    if len(neg_samples) < n_neg_target:
        n_neg_target = len(neg_samples)
        n_pos_target = args.target - n_neg_target
        pos_samples = pos_samples[:n_pos_target]
        print(
            f"[warn] only {len(neg_samples)} negatives; "
            f"using pos={n_pos_target} neg={n_neg_target}",
            flush=True,
        )

    all_samples = pos_samples[:n_pos_target] + neg_samples[:n_neg_target]
    rng.shuffle(all_samples)
    jsonl = args.out / "point_sharegpt.jsonl"
    n = write_jsonl(jsonl, all_samples)
    n_neg_written = sum(1 for s in all_samples if s.meta.get("is_negative"))
    n_multi_samples = sum(
        1 for s in all_samples if (s.meta.get("n_fragments") or 1) > 1 and not s.meta.get("is_negative")
    )
    elapsed = time.time() - t0
    meta_out = {
        "protocol": "a1_center_band_035_multi_rect",
        "target": args.target,
        "n_written": n,
        "n_pos": n - n_neg_written,
        "n_neg": n_neg_written,
        "neg_frac": round(n_neg_written / n, 4) if n else 0.0,
        "neg_frac_target": args.neg_frac,
        "n_multi_fragment_pos": n_multi_samples,
        "n_dense_pages": n_dense_pages,
        "n_sparse_pages": n_sparse_pages,
        "n_skipped_empty": n_skipped_empty,
        "pages_done": pages_done,
        "workers": n_workers,
        "elapsed_s": round(elapsed, 1),
        "pages_per_s": round(pages_done / elapsed, 2) if elapsed > 0 else 0,
        "sparse_page_frac": args.sparse_page_frac,
        "min_area_ratio": args.min_area_ratio,
        "min_chars": args.min_chars,
        "pool": str(args.pool),
        "pool_counts": {k: len(v) for k, v in pool.items() if isinstance(v, list)},
        "noise": use_noise,
    }
    (args.out / "expand_meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(json.dumps(meta_out, indent=2))
    print(f"Wrote {n} → {jsonl}")


if __name__ == "__main__":
    # Required on some platforms for ProcessPool + Playwright
    try:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

#!/usr/bin/env python3
"""Build one objective sample pool (template-diverse). Stages sample later.

  export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
  uv run python data/scripts/build_pool.py --pool-id core_inner --target 20000 --wipe --workers 30
  uv run python data/scripts/build_pool.py --all --workers 30 --wipe
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

from apply_content_pack import (  # noqa: E402
    OCCUPANCY_BANDS,
    PairAllocator,
    densify_static_html,
    fill_from_pool,
    occupancy_layout,
)
from point_ocr.a2_mix import interleave_a2_samples, meta_template_stem  # noqa: E402
from point_ocr.build_point import load_blocks_json  # noqa: E402
from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.noise import apply_screen_noise  # noqa: E402
from point_ocr.chrome.wrap import B1_FAMILY_WEIGHTS, sample_chrome_spec, wrap_page_html  # noqa: E402
from point_ocr.pools.emit import emit_pool_samples  # noqa: E402
from point_ocr.pools.select import OCC_MAX, OCC_MIN, occupancy_in_train_range, text_occupancy_frac  # noqa: E402
from point_ocr.pools.spec import STATIC_HTML_FILES, get_pool_spec, list_pool_ids  # noqa: E402
from point_ocr.synth.render import HtmlRenderSession  # noqa: E402
from point_ocr.synth.window_viewport import sample_window_capture  # noqa: E402

_WORKER: dict[str, Any] = {}


def _fix_playwright_browsers_path() -> None:
    """Cursor injects a sandbox Playwright cache that has no Chromium."""
    home = Path.home() / ".cache" / "ms-playwright"
    cur = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if home.is_dir() and (not cur or "cursor-sandbox-cache" in cur):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(home)


def _worker_init(pool_path: str, chrome_pool_path: str = "") -> None:
    # One Chromium per process; keep BLAS from oversubscribing cores.
    _fix_playwright_browsers_path()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    data = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    _WORKER["pool"] = data.get("pool") or data
    chrome: dict[str, Any] = {"browser_scenes": [], "office_scenes": []}
    if chrome_pool_path and Path(chrome_pool_path).is_file():
        raw = json.loads(Path(chrome_pool_path).read_text(encoding="utf-8"))
        chrome = {
            "browser_scenes": raw.get("browser_scenes") or [],
            "office_scenes": raw.get("office_scenes") or [],
        }
    _WORKER["chrome_pool"] = chrome
    _WORKER["session"] = HtmlRenderSession(wait_until="load").__enter__()


def _dict_to_sample(d: dict[str, Any]) -> PointSample:
    return PointSample(
        sample_id=d["sample_id"],
        image_path=d["image_path"],
        task=d["task"],
        target=d["target"],
        meta=d.get("meta") or {},
    )


def _process_job(job: dict[str, Any]) -> dict[str, Any]:
    # Per-image marker size: workers are separate processes, so apply the range here.
    import point_ocr.marker as _marker

    if job.get("marker_area_min") is not None:
        _marker.MARKER_AREA_FRAC_MIN = float(job["marker_area_min"])
    if job.get("marker_area_max") is not None:
        _marker.MARKER_AREA_FRAC_MAX = float(job["marker_area_max"])
    content_pool = _WORKER["pool"]
    session: HtmlRenderSession = _WORKER["session"]
    pool_id = job["pool_id"]
    spec = get_pool_spec(pool_id)
    page_id = job["page_id"]
    seed = int(job["seed"])
    i = int(job["index"])
    tmpl_path = Path(job["template"])
    template_stem = tmpl_path.stem
    marked_dir = Path(job["marked_dir"])
    render_dir = Path(job["render_dir"])
    filled_dir = Path(job["filled_dir"])
    use_noise = bool(job.get("use_noise", True))
    try:
        html_src = tmpl_path.read_text(encoding="utf-8")
        layout_name = None
        static = spec.static_html or tmpl_path.name in STATIC_HTML_FILES
        occ_band = job.get("occupancy_band")
        extras = int(job.get("extras", 6))
        filled = html_src
        img = None
        meta: dict[str, Any] = {}
        blocks = []
        occ = 0.0
        attempts = 3 if occ_band else 1
        content_pool = job.get("page_pool") or content_pool
        for attempt in range(attempts):
            rng_fill = random.Random(seed + i * 17 + attempt * 91)
            if static and not job.get("fill_static"):
                if occ_band:
                    filled, layout_name = densify_static_html(
                        html_src,
                        content_pool,
                        rng_fill,
                        occupancy_band=str(occ_band),
                        extra_paragraphs=extras,
                    )
                else:
                    filled, layout_name = html_src, None
            else:
                filled, layout_name = fill_from_pool(
                    html_src,
                    content_pool,
                    rng_fill,
                    extra_paragraphs=extras,
                    layout_set="dense",
                    occupancy_band=str(occ_band) if occ_band else None,
                )
            chrome_spec_meta: dict[str, Any] = {
                "chrome_family": "none",
                "chrome_skin": "none",
                "chrome_theme": "light",
            }
            page_html = filled
            if job.get("use_chrome"):
                weights = B1_FAMILY_WEIGHTS
                if job.get("chrome_force"):
                    weights = {"none": 0.0, "browser": 0.55, "office": 0.45}
                ch = sample_chrome_spec(
                    random.Random(seed + i * 13),
                    _WORKER.get("chrome_pool") or {},
                    family_weights=weights,
                )
                page_html = wrap_page_html(filled, ch)
                chrome_spec_meta = ch.to_meta()
            filled_path = filled_dir / f"{page_id}.html"
            filled_path.write_text(page_html, encoding="utf-8")

            cap = sample_window_capture(
                random.Random(seed + i * 41),
                min_css_pixels=spec.extras.get("min_css_pixels"),
                min_aspect=spec.extras.get("min_aspect"),
            )
            meta = session.render(
                filled_path,
                render_dir,
                page_id=page_id,
                viewport=cap.viewport,
                device_scale_factor=cap.device_scale_factor,
                full_page=cap.full_page,
            )
            img = Image.open(meta["image_path"]).convert("RGB")
            if use_noise:
                img = apply_screen_noise(img, rng=random.Random(seed + i * 31))
            blocks = load_blocks_json(Path(meta["blocks_path"]))
            occ = text_occupancy_frac(
                blocks, img.size[0], img.size[1], include_special=True
            )
            if not occ_band or occupancy_in_train_range(occ):
                filled = page_html
                break
            if occ < OCC_MIN:
                extras += 8
                occ_band = "occ_60"
            else:
                extras = max(2, extras // 2)
                occ_band = "occ_30"
        assert img is not None
        if occ_band and not occupancy_in_train_range(occ):
            return {
                "page_id": page_id,
                "pool_id": pool_id,
                "skipped": True,
                "samples": [],
                "error": f"occupancy_out_of_range:{occ:.3f}",
            }
        capture_meta = {
            **cap.to_meta(),
            "render_viewport": list(meta.get("viewport") or cap.viewport),
            "render_device_scale_factor": meta.get("device_scale_factor", cap.device_scale_factor),
            "text_occupancy": round(occ, 4),
            "occupancy_band": occ_band or "static",
            "marker_area_frac_range": [
                float(job.get("marker_area_min") or 0.0),
                float(job.get("marker_area_max") or 0.0),
            ],
            **chrome_spec_meta,
        }
        chrome_bands: list[dict[str, Any]] = []
        chrome_path = Path(meta.get("chrome_path") or "")
        if chrome_path.is_file():
            chrome_bands = json.loads(chrome_path.read_text(encoding="utf-8"))
        samples = emit_pool_samples(
            img,
            blocks,
            pool_id=pool_id,
            page_id=page_id,
            marked_dir=marked_dir,
            seed=seed + i * 7,
            template_stem=template_stem,
            prompt_key=str(job.get("prompt_key") or spec.default_prompt_key),
            layout_name=layout_name,
            capture_meta=capture_meta,
            page_html=filled,
            chrome_bands=chrome_bands,
        )
        return {
            "page_id": page_id,
            "pool_id": pool_id,
            "skipped": False,
            "samples": [asdict(s) for s in samples],
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 — worker must not die
        return {
            "page_id": page_id,
            "pool_id": pool_id,
            "skipped": True,
            "samples": [],
            "error": f"{type(e).__name__}: {e}",
        }


def _default_render_workers() -> int:
    """One Chromium per worker. Leave a single core for OS / the parent process."""
    cpus = os.cpu_count() or 4
    return max(1, cpus - 1)


def _build_one(args: argparse.Namespace, pool_id: str) -> None:
    spec = get_pool_spec(pool_id)
    target = int(args.target) if args.target else spec.default_target
    out = args.out_root / pool_id
    protected_roots = {(ROOT / "data/pools").resolve(), (ROOT / "data/synth").resolve(), (ROOT / "data/processed/a1_v2").resolve()}
    if args.wipe and out.exists():
        resolved = out.resolve()
        if resolved in protected_roots or any(p in resolved.parents for p in protected_roots if p.name == "pools"):
            raise SystemExit(f"Refusing to wipe protected path {out}")
        shutil.rmtree(out)
        print(f"[wipe] removed {out}", flush=True)

    filled_dir = out / "filled_html"
    render_dir = out / "renders"
    marked_dir = out / "marked"
    for d in (filled_dir, render_dir, marked_dir):
        d.mkdir(parents=True, exist_ok=True)

    templates = [args.templates_dir / name for name in spec.templates]
    missing = [p for p in templates if not p.is_file()]
    if missing:
        raise SystemExit(f"missing templates: {missing}")
    if not spec.static_html and not args.content_pool.is_file():
        raise SystemExit(f"Missing content pool {args.content_pool}")

    n_workers = max(1, args.workers)
    # Keep every Chromium busy without queueing hundreds of extra pages past target.
    max_inflight = args.wave_size or max(n_workers * 3, n_workers)
    max_pages = int(args.max_pages)
    use_noise = args.noise and not args.no_noise
    collected: list[PointSample] = []
    pages_done = 0
    pages_failed = 0
    tmpl_rr = 0
    job_i = 0
    t0 = time.time()
    raw_pool: dict[str, Any] = {}
    if args.content_pool.is_file():
        blob = json.loads(args.content_pool.read_text(encoding="utf-8"))
        raw_pool = blob.get("pool") or blob
    allocator = PairAllocator(raw_pool, random.Random(args.seed + 17)) if raw_pool else None

    def _make_job() -> dict[str, Any]:
        nonlocal tmpl_rr, job_i
        tmpl = templates[tmpl_rr % len(templates)]
        tmpl_rr += 1
        page_id = f"{pool_id}__{tmpl.stem}__{job_i}"
        job = {
            "pool_id": pool_id,
            "page_id": page_id,
            "seed": args.seed,
            "index": job_i,
            "template": str(tmpl),
            "marked_dir": str(marked_dir),
            "render_dir": str(render_dir),
            "filled_dir": str(filled_dir),
            "use_noise": use_noise,
            "extras": 6,
            "prompt_key": str(getattr(args, "prompt_key", None) or spec.default_prompt_key),
            "use_chrome": bool(args.chrome),
            "chrome_force": bool(getattr(args, "chrome_force", False)),
            "marker_area_min": getattr(args, "marker_area_min", None),
            "marker_area_max": getattr(args, "marker_area_max", None),
            "fill_static": bool(getattr(args, "fill_static", False)),
        }
        if allocator is not None:
            job["page_pool"] = allocator.take_page(n_per_bucket=int(getattr(args, "pairs_per_bucket", 24)))
        band = OCCUPANCY_BANDS[job_i % len(OCCUPANCY_BANDS)]
        job["occupancy_band"] = band
        job["extras"] = int(occupancy_layout(band)["extra"])
        job_i += 1
        return job

    print(
        f"[start] pool={pool_id} target={target} workers={n_workers} inflight={max_inflight} out={out}",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(args.content_pool), str(args.chrome_pool if args.chrome else "")),
    ) as ex:
        pending: dict[Any, str] = {}

        def _fill_pipeline() -> None:
            remain = max(0, target - len(collected))
            cap = min(max_inflight, max(n_workers, remain + n_workers))
            while len(pending) < cap and job_i < max_pages and len(collected) < target:
                job = _make_job()
                fut = ex.submit(_process_job, job)
                pending[fut] = job["page_id"]

        _fill_pipeline()
        while pending and len(collected) < target:
            done = next(as_completed(pending))
            pending.pop(done)
            pages_done += 1
            try:
                result = done.result()
            except Exception as e:  # noqa: BLE001
                pages_failed += 1
                print(f"[fail] {e}", flush=True)
                _fill_pipeline()
                continue
            if result.get("error"):
                pages_failed += 1
                print(f"[warn] {result.get('page_id')}: {result['error']}", flush=True)
            room = max(0, target - len(collected))
            for row in result.get("samples") or []:
                if room <= 0:
                    break
                collected.append(_dict_to_sample(row))
                room -= 1
            if pages_done % 25 == 0 or len(collected) >= target:
                elapsed = time.time() - t0
                rate = pages_done / elapsed if elapsed > 0 else 0
                samp_rate = len(collected) / elapsed if elapsed > 0 else 0
                print(
                    f"[ok] pages={pages_done} fail={pages_failed} "
                    f"{rate:.2f} pg/s {samp_rate:.1f} samp/s "
                    f"n={len(collected)}/{target} pending={len(pending)}",
                    flush=True,
                )
            if len(collected) < target:
                _fill_pipeline()
        for fut in list(pending.keys()):
            fut.cancel()

    collected = collected[:target]
    rng = random.Random(args.seed)
    collected = interleave_a2_samples(
        collected,
        rng=rng,
        slice_fn=lambda s: str(s.meta.get("pool_id") or pool_id),
        template_fn=lambda s: meta_template_stem(s.meta),
    )
    jsonl = out / "point_sharegpt.jsonl"
    n = write_jsonl(jsonl, collected)
    n_neg = sum(1 for s in collected if s.meta.get("is_negative"))
    elapsed = time.time() - t0
    build_meta = {
        "protocol": "objective_pool_v1",
        "pool_id": pool_id,
        "goal": spec.goal,
        "gt": spec.gt,
        "target": target,
        "n_written": n,
        "n_neg": n_neg,
        "templates": list(spec.templates),
        "static_html": spec.static_html,
        "prompt_key": str(getattr(args, "prompt_key", None) or spec.default_prompt_key),
        "pages_done": pages_done,
        "pages_failed": pages_failed,
        "workers": n_workers,
        "seed": args.seed,
        "out": str(out),
        "elapsed_sec": round(elapsed, 1),
    }
    (out / "pool_meta.json").write_text(
        json.dumps(build_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(build_meta, indent=2), flush=True)
    print(f"Wrote {n} → {jsonl}", flush=True)
    if n < int(0.80 * target):
        raise SystemExit(f"pool {pool_id} too short: got {n} need ~{target}")


def main() -> None:
    _fix_playwright_browsers_path()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-id", action="append", default=[], help="Repeatable. Default: all pools.")
    ap.add_argument("--all", action="store_true", help="Build every registered pool.")
    ap.add_argument("--content-pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--chrome-pool", type=Path, default=ROOT / "data/synth/content_pools/chrome_pool.json")
    ap.add_argument("--chrome", action="store_true", help="Layer-3 window chrome (B1 60/40 none/chrome).")
    ap.add_argument(
        "--chrome-force",
        action="store_true",
        help="With --chrome: never sample family=none (browser/office only).",
    )
    ap.add_argument("--templates-dir", type=Path, default=ROOT / "data/synth/templates")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/pools")
    ap.add_argument("--target", type=int, default=0, help="Samples per pool (0 = spec default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", action="store_true", default=True)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--max-pages", type=int, default=20_000)
    ap.add_argument("--workers", type=int, default=_default_render_workers())
    ap.add_argument("--wave-size", type=int, default=0)
    ap.add_argument("--wipe", action="store_true")
    ap.add_argument(
        "--prompt-key",
        type=str,
        default="",
        help="Override pool default prompt_key (e.g. a2_v3 for Q1).",
    )
    ap.add_argument(
        "--mix-marker-scales",
        action="store_true",
        help="[deprecated] Marker size is now random per image; kept for old scripts.",
    )
    ap.add_argument(
        "--marker-area-min",
        type=float,
        default=None,
        help="Override MARKER_AREA_FRAC_MIN (default 0.0008) for this build.",
    )
    ap.add_argument(
        "--marker-area-max",
        type=float,
        default=None,
        help="Override MARKER_AREA_FRAC_MAX (default 0.002) for this build.",
    )
    ap.add_argument(
        "--fill-static",
        action="store_true",
        help="Replace authored static-template slot text from the content pool.",
    )
    ap.add_argument(
        "--pairs-per-bucket",
        type=int,
        default=12,
        help="Unique bilingual pairs reserved per length bucket for one page.",
    )
    args = ap.parse_args()

    ids = list_pool_ids() if args.all else list(args.pool_id)
    if not ids:
        raise SystemExit(f"pass --pool-id or --all; pools: {list_pool_ids()}")
    for pid in ids:
        get_pool_spec(pid)
        _build_one(args, pid)


if __name__ == "__main__":
    main()

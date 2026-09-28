#!/usr/bin/env python3
"""Build Stage-A1_v2 POINT dataset (doc-only, magenta X marker).

Parallel to legacy A1 ``data/processed/synth`` — never touches it::

  data/processed/a1_v2/{filled_html,renders,marked,point_sharegpt.jsonl,build_meta.json}
  data/splits_a1_v2/{train,val,test}.jsonl

Example:
  export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
  uv run python data/scripts/build_a1_dataset_v2.py --target 15000 --workers 30 --wipe --also-split
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
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
from point_ocr.a1_mix import (  # noqa: E402
    A1_V2_SLICE_FRAC,
    A1_V2_SLICE_TEMPLATES,
    A1_V2_TARGET_DEFAULT,
    a1_v2_quota_counts,
)
from point_ocr.a2_mix import (  # noqa: E402
    interleave_a2_samples,
    meta_template_stem,
    summarize_mix,
    template_stem_from_page_id,
)
from point_ocr.build_point import BlockAnno, build_point_samples_for_page, load_blocks_json  # noqa: E402
from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.filter_qa import filter_block_label  # noqa: E402
from point_ocr.a2_labels import block_fully_in_frame  # noqa: E402
from point_ocr.noise import apply_screen_noise  # noqa: E402
from point_ocr.synth.render import HtmlRenderSession  # noqa: E402
from point_ocr.synth.window_viewport import sample_window_capture  # noqa: E402

_WORKER: dict[str, Any] = {}


def _worker_init(pool_path: str) -> None:
    data = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    _WORKER["pool"] = data.get("pool") or data
    _WORKER["session"] = HtmlRenderSession(wait_until="load").__enter__()


def _dict_to_sample(d: dict[str, Any]) -> PointSample:
    return PointSample(
        sample_id=d["sample_id"],
        image_path=d["image_path"],
        task=d["task"],
        target=d["target"],
        meta=d.get("meta") or {},
    )


def _sample_to_dict(s: PointSample) -> dict[str, Any]:
    return asdict(s)


def _large_blocks(
    blocks: list[BlockAnno],
    page_area: float,
    *,
    image_w: int,
    image_h: int,
    min_area: float = 0.015,
    min_chars: int = 40,
) -> list[BlockAnno]:
    kept: list[BlockAnno] = []
    for b in blocks:
        fr = filter_block_label(b.markdown, tag=(b.extra or {}).get("tag"))
        if not fr.keep:
            continue
        # Window capture may crop blocks; never keep partial-text + full GT.
        if not block_fully_in_frame(b, image_w, image_h):
            continue
        if b.ink_area() / page_area < min_area:
            continue
        if len((b.markdown or "").strip()) < min_chars:
            continue
        kept.append(b)
    kept.sort(key=lambda b: b.ink_area(), reverse=True)
    return kept


def _pick_spread(blocks: list[BlockAnno], n: int, rng: random.Random) -> list[BlockAnno]:
    """Avoid always taking reading-order first blocks."""
    if len(blocks) <= n:
        return list(blocks)
    by_y = sorted(blocks, key=lambda b: b.bbox.y0)
    skip = max(1, len(by_y) // 5)
    pool = by_y[skip:] or by_y
    large = sorted(pool, key=lambda b: b.ink_area(), reverse=True)
    chosen: list[BlockAnno] = []
    for b in large:
        if b not in chosen:
            chosen.append(b)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        rest = [b for b in pool if b not in chosen]
        rng.shuffle(rest)
        chosen.extend(rest[: n - len(chosen)])
    return chosen


def _process_job(job: dict[str, Any]) -> dict[str, Any]:
    pool = _WORKER["pool"]
    session: HtmlRenderSession = _WORKER["session"]
    slice_name = job["slice"]
    page_id = job["page_id"]
    seed = int(job["seed"])
    i = int(job["index"])
    use_noise = bool(job.get("use_noise", True))
    marked_dir = Path(job["marked_dir"])
    render_dir = Path(job["render_dir"])
    filled_dir = Path(job["filled_dir"])
    tmpl_path = Path(job["template"])
    template_stem = tmpl_path.stem

    try:
        html_src = tmpl_path.read_text(encoding="utf-8")
        filled, layout_name = fill_from_pool(
            html_src,
            pool,
            random.Random(seed + i * 17),
            extra_paragraphs=int(job.get("extras", 6)),
            layout_set="dense",
        )
        filled_path = filled_dir / f"{page_id}.html"
        filled_path.write_text(filled, encoding="utf-8")

        cap = sample_window_capture(random.Random(seed + i * 41))
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
        all_boxes = [r for b in blocks for r in b.ink_rects()]
        w, h = img.size
        page_area = float(w * h)
        rng = random.Random(seed + i * 7)
        samples: list[PointSample] = []
        capture_meta = {
            **cap.to_meta(),
            "render_viewport": list(meta.get("viewport") or cap.viewport),
            "render_device_scale_factor": meta.get("device_scale_factor", cap.device_scale_factor),
        }

        if slice_name == "long_center":
            kept = _pick_spread(_large_blocks(blocks, page_area, image_w=w, image_h=h), 5, rng)
            batch = build_point_samples_for_page(
                img,
                kept,
                page_id=page_id,
                out_image_dir=marked_dir,
                r_min=1,
                r_max=2,
                n_negatives=0,
                seed=seed + i,
                avoid_boxes=all_boxes,
                coverage="center_band",
                marker="x45",
                meta_extra={
                    "a1_slice": "long_center",
                    "layout": layout_name,
                    "prompt_key": "a1_v2",
                    **capture_meta,
                },
            )
            samples = [s for s in batch if not s.meta.get("is_negative")]

        elif slice_name == "empty_neg":
            n_neg = int(job.get("n_negatives", 8))
            batch = build_point_samples_for_page(
                img,
                [],
                page_id=page_id,
                out_image_dir=marked_dir,
                r_min=1,
                r_max=1,
                n_negatives=n_neg,
                seed=seed + i,
                avoid_boxes=all_boxes,
                coverage="center_band",
                marker="x45",
                meta_extra={
                    "a1_slice": "empty_neg",
                    "layout": layout_name,
                    "prompt_key": "a1_v2",
                    **capture_meta,
                },
            )
            samples = [s for s in batch if s.meta.get("is_negative")]

        else:
            return {
                "page_id": page_id,
                "slice": slice_name,
                "skipped": True,
                "samples": [],
                "error": f"unknown slice {slice_name}",
            }

        for s in samples:
            s.meta["a1_slice"] = slice_name
            s.meta["marker"] = "x45"
            s.meta["prompt_key"] = "a1_v2"
            s.meta["template_stem"] = template_stem
            s.meta.update(capture_meta)

        return {
            "page_id": page_id,
            "slice": slice_name,
            "skipped": len(samples) == 0,
            "samples": [_sample_to_dict(s) for s in samples],
            "n": len(samples),
        }
    except Exception as e:
        return {
            "page_id": page_id,
            "slice": slice_name,
            "skipped": True,
            "samples": [],
            "error": f"{type(e).__name__}: {e}",
        }


def _default_render_workers() -> int:
    """Use most CPU cores; each worker owns a Chromium (leave 2 cores for OS/IO)."""
    cpus = os.cpu_count() or 4
    return max(1, cpus - 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--templates-dir", type=Path, default=ROOT / "data/synth/templates")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/a1_v2")
    ap.add_argument("--target", type=int, default=A1_V2_TARGET_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", action="store_true", default=True)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--max-pages", type=int, default=12_000)
    ap.add_argument(
        "--workers",
        type=int,
        default=_default_render_workers(),
        help="Parallel Playwright workers (default: cpu_count-2)",
    )
    ap.add_argument(
        "--wave-size",
        type=int,
        default=0,
        help="Max in-flight pages (default workers*12)",
    )
    ap.add_argument("--wipe", action="store_true")
    ap.add_argument("--split-out", type=Path, default=ROOT / "data/splits_a1_v2")
    ap.add_argument("--also-split", action="store_true")
    args = ap.parse_args()

    use_noise = args.noise and not args.no_noise
    n_workers = max(1, args.workers)
    max_inflight = args.wave_size or (n_workers * 12)

    if not args.pool.is_file():
        raise SystemExit(f"Missing pool {args.pool}")

    quotas = a1_v2_quota_counts(args.target)
    templates: dict[str, list[Path]] = {}
    for sl, names in A1_V2_SLICE_TEMPLATES.items():
        paths = [args.templates_dir / n for n in names if (args.templates_dir / n).is_file()]
        if not paths:
            raise SystemExit(f"No templates for slice {sl}")
        templates[sl] = paths

    filled_dir = args.out / "filled_html"
    render_dir = args.out / "renders"
    marked_dir = args.out / "marked"

    forbidden = {ROOT / "data/processed/synth", ROOT / "data/splits", ROOT / "data/processed/a2"}
    if args.wipe and args.out.exists():
        if args.out.resolve() in {p.resolve() for p in forbidden}:
            raise SystemExit(f"Refusing to wipe protected path: {args.out}")
        shutil.rmtree(args.out)
        print(f"[wipe] removed {args.out}", flush=True)

    for d in (filled_dir, render_dir, marked_dir):
        d.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[PointSample]] = {k: [] for k in quotas}
    tmpl_rr: dict[str, int] = {k: 0 for k in quotas}
    in_flight: dict[str, int] = {k: 0 for k in quotas}
    pages_done = 0
    pages_failed = 0
    next_i = 0
    t0 = time.time()

    print(
        f"[start] A1_v2 target={args.target} quotas={quotas} workers={n_workers} "
        f"inflight={max_inflight} out={args.out}",
        flush=True,
    )

    def _need(slice_name: str) -> int:
        reserved = in_flight[slice_name] * 4
        return max(0, quotas[slice_name] - len(buckets[slice_name]) - reserved)

    def _enough() -> bool:
        return all(len(buckets[k]) >= quotas[k] for k in quotas)

    def _pick_slice() -> str | None:
        needy = [k for k in quotas if _need(k) > 0]
        if not needy:
            return None
        weights = [max(1, quotas[k] - len(buckets[k])) for k in needy]
        return random.Random(args.seed + next_i * 13).choices(needy, weights=weights, k=1)[0]

    def _make_job(slice_name: str, index: int) -> dict[str, Any]:
        tmpls = templates[slice_name]
        tmpl = tmpls[tmpl_rr[slice_name] % len(tmpls)]
        tmpl_rr[slice_name] += 1
        return {
            "slice": slice_name,
            "template": str(tmpl),
            "page_id": f"{slice_name}__{tmpl.stem}__{index}",
            "index": index,
            "seed": args.seed,
            "filled_dir": str(filled_dir),
            "render_dir": str(render_dir),
            "marked_dir": str(marked_dir),
            "use_noise": use_noise,
            "extras": 6,
            "n_negatives": 10 if slice_name == "empty_neg" else 0,
        }

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(args.pool.resolve()),),
    ) as ex:
        pending: dict[Any, str] = {}

        def _fill_pipeline() -> None:
            nonlocal next_i
            while len(pending) < max_inflight and not _enough() and next_i < args.max_pages:
                slice_name = _pick_slice()
                if slice_name is None:
                    break
                job = _make_job(slice_name, next_i)
                fut = ex.submit(_process_job, job)
                pending[fut] = slice_name
                in_flight[slice_name] += 1
                next_i += 1

        _fill_pipeline()
        while pending and not _enough():
            fut = next(as_completed(pending.keys()))
            slice_hint = pending.pop(fut, None)
            if slice_hint:
                in_flight[slice_hint] = max(0, in_flight[slice_hint] - 1)
            pages_done += 1
            try:
                result = fut.result()
            except Exception as e:
                pages_failed += 1
                print(f"[warn] future failed ({slice_hint}): {e}", flush=True)
                _fill_pipeline()
                continue
            if result.get("error"):
                pages_failed += 1
                if pages_failed <= 20 or pages_failed % 50 == 0:
                    print(f"[warn] {result.get('page_id')}: {result['error']}", flush=True)
            sl = str(result.get("slice") or slice_hint or "")
            if sl in buckets:
                room = max(0, quotas[sl] - len(buckets[sl]))
                for row in result.get("samples") or []:
                    if room <= 0:
                        break
                    s = _dict_to_sample(row)
                    s.meta["a1_slice"] = sl
                    s.meta.setdefault("marker", "v2")
                    if not s.meta.get("template_stem"):
                        s.meta["template_stem"] = template_stem_from_page_id(
                            str(s.meta.get("page_id") or "")
                        )
                    buckets[sl].append(s)
                    room -= 1

            if pages_done % 25 == 0 or _enough():
                elapsed = time.time() - t0
                filled = {k: len(v) for k, v in buckets.items()}
                total = sum(filled.values())
                rate = pages_done / elapsed if elapsed > 0 else 0
                samp_rate = total / elapsed if elapsed > 0 else 0
                print(
                    f"[ok] pages={pages_done} fail={pages_failed} "
                    f"{rate:.2f} pg/s {samp_rate:.1f} samp/s "
                    f"n={total}/{args.target} pending={len(pending)} filled={filled}",
                    flush=True,
                )

            if not _enough():
                _fill_pipeline()
            if _enough():
                break

        for fut in list(pending.keys()):
            fut.cancel()

    short = {k: quotas[k] - len(buckets[k]) for k in quotas if len(buckets[k]) < quotas[k]}
    if short:
        print(f"[warn] under-quota slices: {short}", flush=True)

    all_samples: list[PointSample] = []
    for k in quotas:
        all_samples.extend(buckets[k][: quotas[k]])

    rng = random.Random(args.seed)
    all_samples = interleave_a2_samples(
        all_samples,
        rng=rng,
        slice_fn=lambda s: str(s.meta.get("a1_slice") or "?"),
        template_fn=lambda s: meta_template_stem(s.meta),
    )

    jsonl = args.out / "point_sharegpt.jsonl"
    n = write_jsonl(jsonl, all_samples)
    mix = summarize_mix(
        [
            {
                "metadata": {
                    **s.meta,
                    "a2_slice": s.meta.get("a1_slice"),  # reuse summarizer key
                }
            }
            for s in all_samples
        ]
    )
    # rewrite mix keys for clarity
    mix["by_slice"] = {
        str(s.meta.get("a1_slice")): sum(
            1 for x in all_samples if x.meta.get("a1_slice") == s.meta.get("a1_slice")
        )
        for s in all_samples
    }
    # simpler recount
    from collections import Counter

    mix["by_slice"] = dict(sorted(Counter(s.meta.get("a1_slice") for s in all_samples).items()))
    mix["by_template"] = dict(
        sorted(Counter(meta_template_stem(s.meta) for s in all_samples).items())
    )
    n_neg = sum(1 for s in all_samples if s.meta.get("is_negative"))
    elapsed = time.time() - t0
    build_meta = {
        "protocol": "a1_v2_window_capture_diverse_res",
        "capture": "focused_window_viewport_mix",
        "marker_spec": {
            "tag": "x45",
            "rotation_deg": 45.0,
            "cross_half_length_px": 34,
            "cross_width_px": 3,
            "ring_radius_px": 28,
            "fixed_magenta": True,
            "scales_with_short_side": False,
        },
        "target": args.target,
        "n_written": n,
        "n_neg": n_neg,
        "neg_frac": round(n_neg / n, 4) if n else 0.0,
        "quotas": quotas,
        "slice_frac": dict(A1_V2_SLICE_FRAC),
        "filled": {k: len(buckets[k]) for k in quotas},
        "pages_done": pages_done,
        "pages_failed": pages_failed,
        "workers": n_workers,
        "seed": args.seed,
        "out": str(args.out),
        "elapsed_sec": round(elapsed, 1),
        "mix": mix,
    }
    (args.out / "build_meta.json").write_text(
        json.dumps(build_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(build_meta, indent=2), flush=True)
    print(f"Wrote {n} → {jsonl}", flush=True)

    if n < int(0.95 * args.target):
        raise SystemExit(
            f"A1_v2 build too short: got {n} need ~{args.target}; raise --max-pages"
        )

    if args.also_split:
        args.split_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "data/scripts/split_train_val_test.py"),
            "--input",
            str(jsonl),
            "--out-dir",
            str(args.split_out),
            "--seed",
            str(args.seed),
            "--stratify-key",
            "a1_slice",
            "--interleave-template",
        ]
        print("[split]", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

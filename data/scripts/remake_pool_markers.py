#!/usr/bin/env python3
"""Re-stamp existing pool samples with the current MARKER_SPEC (no re-render).

Reads each pool's ``point_sharegpt.jsonl`` + clean ``renders/<page_id>.png``,
re-applies screen noise with the recovered build seed, draws ``draw_crosshair``,
overwrites marked JPEGs, and rewrites the JSONL (marker tag + current prompt).

  uv run python data/scripts/remake_pool_markers.py --pools-root data/pools --workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from point_ocr.marker import (  # noqa: E402
    CURRENT_MARKER_TAG,
    MARKER_AREA_FRAC_MAX,
    MARKER_AREA_FRAC_MIN,
    MARKER_SPEC,
    draw_crosshair,
    sample_marker_area_frac,
    spec_for_image,
)
from point_ocr.noise import apply_screen_noise  # noqa: E402
from point_ocr.prompts import get_prompt  # noqa: E402

_PAGE_INDEX_RE = re.compile(r"__(\d+)$")


def _page_index(page_id: str) -> int | None:
    m = _PAGE_INDEX_RE.search(page_id)
    return int(m.group(1)) if m else None


def _remake_one(job: dict[str, Any]) -> dict[str, Any]:
    try:
        render = Path(job["render"])
        out_img = Path(job["image"])
        if not render.is_file():
            return {"ok": False, "error": f"missing render {render}", "sample_id": job["sample_id"]}
        img = Image.open(render).convert("RGB")
        if job.get("use_noise"):
            img = apply_screen_noise(img, rng=__import__("random").Random(int(job["noise_seed"])))
        pt = job["point"]
        frac = job.get("marker_area_frac")
        if frac is None:
            frac = sample_marker_area_frac(__import__("random").Random(int(job.get("noise_seed") or 0)))
        marked = draw_crosshair(
            img, float(pt[0]), float(pt[1]), spec_for_image(*img.size, area_frac=float(frac))
        )
        out_img.parent.mkdir(parents=True, exist_ok=True)
        marked.save(out_img, format="JPEG", quality=92)
        return {"ok": True, "sample_id": job["sample_id"], "error": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "sample_id": job.get("sample_id"), "error": f"{type(e).__name__}: {e}"}


def remake_pool(
    pool_dir: Path,
    *,
    build_seed: int,
    prompt_key: str,
    workers: int,
    use_noise: bool,
) -> dict[str, Any]:
    jsonl_path = pool_dir / "point_sharegpt.jsonl"
    if not jsonl_path.is_file():
        raise SystemExit(f"missing {jsonl_path}")
    render_dir = pool_dir / "renders"
    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    prompt = get_prompt("POINT", prompt_key=prompt_key)
    jobs: list[dict[str, Any]] = []
    missing_idx = 0
    for r in rows:
        meta = r.get("metadata") or {}
        page_id = str(meta.get("page_id") or "")
        point = meta.get("point")
        images = r.get("images") or []
        if not page_id or not point or len(point) < 2 or not images:
            continue
        idx = _page_index(page_id)
        if idx is None:
            missing_idx += 1
            noise_seed = build_seed
        else:
            noise_seed = build_seed + idx * 31
        jobs.append(
            {
                "sample_id": meta.get("sample_id") or page_id,
                "render": str(render_dir / f"{page_id}.png"),
                "image": str(images[0]),
                "point": point,
                "noise_seed": noise_seed,
                "use_noise": use_noise,
                "marker_area_frac": meta.get("marker_area_frac"),
            }
        )

    t0 = time.time()
    ok = 0
    fail = 0
    errors: list[str] = []
    print(
        f"[start] {pool_dir.name} n={len(jobs)} workers={workers} marker={CURRENT_MARKER_TAG}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_remake_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res["ok"]:
                ok += 1
            else:
                fail += 1
                if len(errors) < 8:
                    errors.append(f"{res.get('sample_id')}: {res.get('error')}")
            if i % 1000 == 0 or i == len(futs):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                print(
                    f"  [{pool_dir.name}] {i}/{len(futs)} ok={ok} fail={fail} {rate:.1f} samp/s",
                    flush=True,
                )

    # Rewrite JSONL: marker tag + prompt text (compose also rewrites prompt).
    out_tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with out_tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            meta = dict(r.get("metadata") or {})
            meta["marker"] = CURRENT_MARKER_TAG
            r["metadata"] = meta
            msgs = r.get("messages") or []
            if msgs and isinstance(msgs[0], dict):
                content = msgs[0].get("content") or ""
                if content.startswith("<image>"):
                    msgs[0]["content"] = f"<image>{prompt}"
                else:
                    msgs[0]["content"] = prompt
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_tmp.replace(jsonl_path)

    meta_out = {
        "pool": pool_dir.name,
        "n": len(rows),
        "ok": ok,
        "fail": fail,
        "missing_page_index": missing_idx,
        "marker": CURRENT_MARKER_TAG,
        "marker_spec_template": MARKER_SPEC.to_dict(),
        "marker_area_frac_range": [MARKER_AREA_FRAC_MIN, MARKER_AREA_FRAC_MAX],
        "prompt_key": prompt_key,
        "build_seed": build_seed,
        "use_noise": use_noise,
        "elapsed_sec": round(time.time() - t0, 2),
        "errors_sample": errors,
    }
    (pool_dir / "remake_marker_meta.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta_out, indent=2, ensure_ascii=False), flush=True)
    return meta_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools-root", type=Path, default=ROOT / "data/pools")
    ap.add_argument(
        "--pool-id",
        action="append",
        default=[],
        help="Pool id(s); default = all dirs with point_sharegpt.jsonl",
    )
    ap.add_argument("--seed", type=int, default=42, help="Original build_pool --seed")
    ap.add_argument("--prompt-key", type=str, default="a2_v2")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-noise", action="store_true")
    args = ap.parse_args()

    if args.pool_id:
        pools = [args.pools_root / p for p in args.pool_id]
    else:
        pools = sorted(
            p for p in args.pools_root.iterdir() if (p / "point_sharegpt.jsonl").is_file()
        )
    if not pools:
        raise SystemExit(f"no pools under {args.pools_root}")

    summaries = []
    for pool in pools:
        summaries.append(
            remake_pool(
                pool,
                build_seed=args.seed,
                prompt_key=args.prompt_key,
                workers=args.workers,
                use_noise=not args.no_noise,
            )
        )
    failed = sum(s["fail"] for s in summaries)
    print(
        f"[done] pools={len(summaries)} total_ok={sum(s['ok'] for s in summaries)} total_fail={failed}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

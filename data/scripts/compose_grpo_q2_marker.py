#!/usr/bin/env python3
"""Compose GRPO_2 split: preferred probe rows + remade dual-size markers.

  uv run python data/scripts/compose_grpo_q2_marker.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import load_jsonl  # noqa: E402
from point_ocr.grpo_data import (  # noqa: E402
    _meta,
    _tag_row,
    interleave_stage,
    select_sliced_rows,
    val_slice,
)
from point_ocr.marker import (  # noqa: E402
    CURRENT_MARKER_TAG,
    MARKER_AREA_FRAC_LEGACY,
    MARKER_AREA_SCALE_LARGE,
    MARKER_AREA_SCALE_SMALL,
    area_frac_for_scale,
    draw_crosshair,
    spec_for_image,
)
from point_ocr.pools.compose import mix_summary, write_rows  # noqa: E402

DEFAULT_RECIPE = ROOT / "data/recipes/grpo_q2_marker.yaml"
POOLS_Q = ROOT / "data" / "pools_q"
POOLS_REAL = ROOT / "data" / "pools_real"


def _load_probe(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def resolve_unmarked(meta: dict[str, Any]) -> Path | None:
    """Locate the unmarked page for a ShareGPT POINT row."""
    pool = str(meta.get("pool_id") or "")
    page_id = str(meta.get("page_id") or "")
    cands: list[Path] = []
    if pool and page_id:
        cands.append(POOLS_Q / pool / "renders" / f"{page_id}.png")
        cands.append(POOLS_REAL / pool / "renders" / f"{page_id}.png")
        # real_labeled pages drop the pool prefix in renders/
        if page_id.startswith(f"{pool}__"):
            short = page_id[len(pool) + 2 :]
            cands.append(POOLS_REAL / pool / "renders" / f"{short}.png")
            cands.append(POOLS_Q / pool / "renders" / f"{short}.png")
    src = str(meta.get("source_path") or "").strip()
    if src:
        cands.append(Path(src))
    for p in cands:
        if p.is_file():
            return p
    return None


def _safe_stem(sample_id: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", sample_id)[:180]


def remake_marked(
    row: dict[str, Any],
    *,
    area_scale: float,
    marked_root: Path,
    jpeg_quality: int,
) -> dict[str, Any]:
    meta = dict(_meta(row))
    point = meta.get("point")
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise ValueError(f"missing point for {meta.get('sample_id')}")
    src = resolve_unmarked(meta)
    if src is None:
        raise FileNotFoundError(f"no unmarked render for {meta.get('sample_id')}")
    img = Image.open(src).convert("RGB")
    w, h = img.size
    spec = spec_for_image(w, h, area_frac=area_frac_for_scale(area_scale))
    stamped = draw_crosshair(img, float(point[0]), float(point[1]), spec)
    pool = str(meta.get("pool_id") or "misc")
    out_dir = marked_root / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = str(meta.get("sample_id") or "sample")
    scale_tag = f"a{area_scale:g}".replace(".", "p")
    name = f"{_safe_stem(sid)}__{scale_tag}.jpg"
    out_path = out_dir / name
    stamped.save(out_path, format="JPEG", quality=jpeg_quality)

    out = dict(row)
    out["images"] = [str(out_path.resolve())]
    out_meta = dict(meta)
    out_meta["marker"] = CURRENT_MARKER_TAG
    out_meta["marker_area_scale"] = float(area_scale)
    out_meta["marker_area_frac"] = float(MARKER_AREA_FRAC_LEGACY * area_scale)
    out_meta["marker_source_render"] = str(src.resolve())
    out_meta["image_w"] = w
    out_meta["image_h"] = h
    out["metadata"] = out_meta
    return out


def assign_area_scales(
    scenes: list[str],
    *,
    hard_scenes: set[str],
    large: float,
    small: float,
    small_frac: float,
    rng: random.Random,
) -> list[float]:
    hard_idx = [i for i, sc in enumerate(scenes) if sc in hard_scenes]
    scales = [large] * len(scenes)
    if not hard_idx:
        return scales
    rng.shuffle(hard_idx)
    n_small = int(round(len(hard_idx) * small_frac))
    for i in hard_idx[:n_small]:
        scales[i] = small
    for i in hard_idx[n_small:]:
        scales[i] = large
    return scales


def select_probe_ids(
    probe: list[dict[str, Any]],
    *,
    n: int,
    min_max: float,
    max_mean: float,
    fallback_min_max: float,
    fallback_max_mean: float,
    by_id: dict[str, dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], str]:
    def pool(min_max_r: float, max_mean_r: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in probe:
            if rec.get("is_negative") or str(rec.get("scene") or "") == "empty":
                continue
            if rec.get("verdict") != "signal":
                continue
            if float(rec.get("reward_max") or 0.0) < min_max_r:
                continue
            if float(rec.get("reward_mean") or 1.0) >= max_mean_r:
                continue
            sid = str(rec.get("sample_id") or "")
            row = by_id.get(sid)
            if row is None:
                continue
            if resolve_unmarked(_meta(row)) is None:
                continue
            out.append(rec)
        return out

    preferred = pool(min_max, max_mean)
    mode = f"maxR>={min_max}&meanR<{max_mean}"
    cand = preferred
    if len(cand) < n:
        cand = pool(fallback_min_max, fallback_max_mean)
        mode = f"fallback maxR>={fallback_min_max}&meanR<{fallback_max_mean}"
    if len(cand) < n:
        raise SystemExit(f"only {len(cand)} eligible probe rows; need {n} ({mode})")
    picked = rng.sample(cand, n)
    return picked, mode


def compose(recipe: dict[str, Any]) -> dict[str, Any]:
    seed = int(recipe.get("seed") or 42)
    stage = str(recipe.get("name") or "grpo_q2_marker")
    src_dir = ROOT / str(recipe.get("from") or "data/splits_stage_q1_withreal")
    probe_path = ROOT / str(recipe.get("probe") or "data/splits_grpo_q1_signal/probe_rescored.jsonl")
    out_dir = ROOT / str(recipe.get("out") or "data/splits_grpo_q2_marker")
    marked_root = ROOT / str(recipe.get("marked_root") or "data/marked_grpo_q2")
    n_train = int(recipe.get("n_train") or 200)
    large = float(recipe.get("area_scale_large") or MARKER_AREA_SCALE_LARGE)
    small = float(recipe.get("area_scale_small") or MARKER_AREA_SCALE_SMALL)
    small_frac = float(recipe.get("small_frac_hard") or 0.70)
    hard_scenes = {str(s) for s in (recipe.get("hard_scenes") or ["multi_frag", "semantic_group"])}
    jpeg_quality = int(recipe.get("jpeg_quality") or 92)

    train_src = load_jsonl(src_dir / "train.jsonl")
    by_id = {str(_meta(r).get("sample_id") or ""): r for r in train_src}
    probe = _load_probe(probe_path)
    rng = random.Random(seed)
    picked, mode = select_probe_ids(
        probe,
        n=n_train,
        min_max=float(recipe.get("min_max_reward") or 0.7),
        max_mean=float(recipe.get("max_mean_reward") or 0.85),
        fallback_min_max=float(recipe.get("fallback_min_max_reward") or 0.5),
        fallback_max_mean=float(recipe.get("fallback_max_mean_reward") or 0.7),
        by_id=by_id,
        rng=rng,
    )

    scenes = [str(r.get("scene") or "multi_frag") for r in picked]
    scales = assign_area_scales(
        scenes,
        hard_scenes=hard_scenes,
        large=large,
        small=small,
        small_frac=small_frac,
        rng=random.Random(seed + 3),
    )

    train: list[dict[str, Any]] = []
    scale_counts: Counter[str] = Counter()
    for rec, area_scale in zip(picked, scales, strict=True):
        sid = str(rec.get("sample_id") or "")
        row = by_id[sid]
        scene = str(rec.get("scene") or val_slice(row) or "multi_frag")
        remade = remake_marked(
            row, area_scale=area_scale, marked_root=marked_root, jpeg_quality=jpeg_quality
        )
        tagged = _tag_row(remade, scene=scene, stage=stage)
        tagged["metadata"]["probe_verdict"] = rec.get("verdict")
        tagged["metadata"]["probe_reward_mean"] = rec.get("reward_mean")
        tagged["metadata"]["probe_reward_std"] = rec.get("reward_std")
        tagged["metadata"]["probe_reward_max"] = rec.get("reward_max")
        train.append(tagged)
        scale_counts[f"{scene}:{area_scale:g}"] += 1

    train = interleave_stage(train, random.Random(seed + 4))

    n_val = int(recipe.get("n_val") or 0)
    val_mix = {str(k): float(v) for k, v in (recipe.get("val_mix") or {}).items()}
    val: list[dict[str, Any]] = []
    if n_val and val_mix and (src_dir / "val.jsonl").is_file():
        chrome = recipe.get("empty_chrome_frac")
        raw_val = select_sliced_rows(
            load_jsonl(src_dir / "val.jsonl"),
            n_val,
            val_mix,
            random.Random(seed + 1),
            slice_fn=val_slice,
            stage=stage,
            empty_chrome_frac=float(chrome) if chrome is not None else None,
        )
        val_scenes = [str(_meta(r).get("grpo_scene") or val_slice(r) or "regular") for r in raw_val]
        val_scales = assign_area_scales(
            val_scenes,
            hard_scenes=hard_scenes,
            large=large,
            small=small,
            small_frac=small_frac,
            rng=random.Random(seed + 5),
        )
        for row, area_scale, scene in zip(raw_val, val_scales, val_scenes, strict=True):
            # empty / regular still get large (new default); hard get mix.
            if scene not in hard_scenes:
                area_scale = large
            try:
                remade = remake_marked(
                    row, area_scale=area_scale, marked_root=marked_root, jpeg_quality=jpeg_quality
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"[val skip] {e}", flush=True)
                continue
            tagged = _tag_row(remade, scene=scene, stage=stage)
            val.append(tagged)
        val = interleave_stage(val, random.Random(seed + 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    hard_n = sum(1 for sc in scenes if sc in hard_scenes)
    hard_small = sum(
        1
        for sc, s in zip(scenes, scales, strict=True)
        if sc in hard_scenes and abs(s - small) < 1e-9
    )
    meta = {
        "name": stage,
        "from": str(src_dir),
        "probe": str(probe_path),
        "marked_root": str(marked_root),
        "select_mode": mode,
        "n_probe": len(probe),
        "n_train": write_rows(out_dir / "train.jsonl", train),
        "n_val": write_rows(out_dir / "val.jsonl", val) if val else 0,
        "val_mix": val_mix,
        "area_scale_large": large,
        "area_scale_small": small,
        "small_frac_hard": small_frac,
        "hard_scenes": sorted(hard_scenes),
        "train_scale_counts": dict(scale_counts),
        "train_hard_small_frac": (hard_small / hard_n) if hard_n else None,
        "train": mix_summary(train),
        "val": mix_summary(val) if val else {},
        "train_by_scene": dict(Counter(str(_meta(r).get("grpo_scene")) for r in train)),
        "val_by_scene": dict(Counter(str(_meta(r).get("grpo_scene")) for r in val)),
        "marker_tag": CURRENT_MARKER_TAG,
    }
    (out_dir / "split_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    args = ap.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise SystemExit(f"recipe {args.recipe} is not a mapping")
    meta = compose(recipe)
    keys = (
        "name",
        "select_mode",
        "n_train",
        "n_val",
        "train_by_scene",
        "val_by_scene",
        "train_hard_small_frac",
        "train_scale_counts",
        "marked_root",
    )
    print(json.dumps({k: meta[k] for k in keys if k in meta}, indent=2), flush=True)
    print(f"wrote {ROOT / str(recipe.get('out') or 'data/splits_grpo_q2_marker')}", flush=True)


if __name__ == "__main__":
    main()

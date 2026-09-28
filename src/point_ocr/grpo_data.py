"""Select GRPO prompts from a POINT ShareGPT split.

Hard scenes after q1_withreal SFT: multi-fragment blocks, short semantic
groups, and empty-on-blank (plus table/image empty).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from point_ocr.dataset_format import load_jsonl
from point_ocr.pools.compose import interleave_stage, mix_summary, write_rows
from point_ocr.pools.spec import quota_from_frac

EMPTY_POOLS = frozenset({"empty_clear", "empty_special", "empty_boundary"})
GRPO_SCENES = ("multi_frag", "semantic_group", "empty")
VAL_SLICES = ("regular", "multi_frag", "semantic_group", "empty")
# Real labeled units shorter than this are treated as short-group prompts.
REAL_SHORT_CHARS = 80


def assistant_text(row: dict[str, Any]) -> str:
    for m in row.get("messages") or []:
        if m.get("role") == "user":
            continue
        if m.get("role") == "assistant":
            return str(m.get("content") or "")
    return ""


def user_text(row: dict[str, Any]) -> str:
    for m in row.get("messages") or []:
        if m.get("role") != "user":
            continue
        raw = m.get("content") or ""
        if isinstance(raw, str) and raw.startswith("<image>"):
            return raw[len("<image>") :]
        return str(raw)
    return ""


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") or row.get("meta") or {}


def is_empty_target(row: dict[str, Any]) -> bool:
    meta = _meta(row)
    if meta.get("is_negative"):
        return True
    return not assistant_text(row).strip()


def chrome_bucket(row: dict[str, Any]) -> str:
    fam = str(_meta(row).get("chrome_family") or "none").strip().lower()
    return "none" if fam in {"", "none"} else "chrome"


def grpo_scene(row: dict[str, Any]) -> str | None:
    """Map a SFT row to a GRPO hard scene, or None if it is not one."""
    meta = _meta(row)
    pool = str(meta.get("pool_id") or "")
    gt = assistant_text(row).strip()
    n_frag = int(meta.get("n_fragments") or 1)
    n_box = len(meta.get("bboxes") or [])
    if is_empty_target(row) or pool in EMPTY_POOLS:
        return "empty"
    if pool == "multi_frag" or n_frag > 1 or n_box > 1:
        return "multi_frag"
    if pool == "semantic_group":
        return "semantic_group"
    if pool == "real_labeled" and 0 < len(gt) < REAL_SHORT_CHARS:
        return "semantic_group"
    return None


def val_slice(row: dict[str, Any]) -> str | None:
    """Held-out readout slice: hard GRPO scenes plus ordinary non-empty blocks."""
    tagged = str(_meta(row).get("grpo_scene") or "").strip()
    if tagged in VAL_SLICES:
        return tagged
    scene = grpo_scene(row)
    if scene is not None:
        return scene
    if assistant_text(row).strip():
        return "regular"
    return None


def _page_key(row: dict[str, Any]) -> str:
    meta = _meta(row)
    return str(meta.get("page_id") or meta.get("sample_id") or id(row))


def pick_unique_page_first(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    *,
    chrome_frac: float | None = None,
) -> list[dict[str, Any]]:
    """Sample ``n`` rows, preferring unseen pages, then filling leftovers."""
    if n <= 0 or not rows:
        return []
    pool = list(rows)
    rng.shuffle(pool)
    if chrome_frac is not None:
        chrome = [r for r in pool if chrome_bucket(r) == "chrome"]
        none = [r for r in pool if chrome_bucket(r) != "chrome"]
        n_chrome = min(len(chrome), int(round(n * chrome_frac)))
        n_none = min(len(none), n - n_chrome)
        # If one side is short, take the rest from the other.
        if n_chrome + n_none < n:
            extra = n - n_chrome - n_none
            if len(chrome) - n_chrome >= extra:
                n_chrome += extra
            else:
                n_none = min(len(none), n_none + extra)
        picked = _unique_then_fill(chrome, n_chrome, rng) + _unique_then_fill(none, n_none, rng)
        rng.shuffle(picked)
        return picked[:n]
    return _unique_then_fill(pool, n, rng)


def _unique_then_fill(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    order = list(rows)
    rng.shuffle(order)
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    leftover: list[dict[str, Any]] = []
    for row in order:
        if len(picked) >= n:
            break
        key = _page_key(row)
        if key not in seen:
            picked.append(row)
            seen.add(key)
        else:
            leftover.append(row)
    for row in leftover:
        if len(picked) >= n:
            break
        picked.append(row)
    return picked[:n]


def _tag_row(row: dict[str, Any], *, scene: str, stage: str) -> dict[str, Any]:
    out = dict(row)
    meta = dict(_meta(row))
    meta["grpo_scene"] = scene
    meta["stage"] = stage
    meta["is_negative"] = bool(is_empty_target(row))
    out["metadata"] = meta
    return out


def is_real_row(row: dict[str, Any]) -> bool:
    return str(_meta(row).get("pool_id") or "") == "real_labeled"


def select_grpo_rows(
    rows: list[dict[str, Any]],
    n: int,
    mix: dict[str, float],
    rng: random.Random,
    *,
    stage: str,
    empty_pools: dict[str, float] | None = None,
    empty_chrome_frac: float | None = None,
    keep_all_real: bool = False,
) -> list[dict[str, Any]]:
    """Draw ``n`` tagged rows according to scene mix, without replacement.

    If ``keep_all_real``, every GRPO-eligible ``real_labeled`` row is kept
    first; remaining slots are filled from synthetic rows.
    """
    if keep_all_real:
        reals = [r for r in rows if is_real_row(r) and grpo_scene(r) is not None]
        tagged_reals = [
            _tag_row(r, scene=grpo_scene(r) or "semantic_group", stage=stage) for r in reals
        ]
        if len(tagged_reals) >= n:
            rng.shuffle(tagged_reals)
            return tagged_reals[:n]
        remaining = n - len(tagged_reals)
        have: dict[str, int] = defaultdict(int)
        for row in tagged_reals:
            have[str(_meta(row).get("grpo_scene") or "")] += 1
        quotas = quota_from_frac(mix, n)
        need = {scene: max(0, int(quotas.get(scene, 0)) - int(have.get(scene, 0))) for scene in mix}
        assigned = sum(need.values())
        if assigned != remaining:
            if assigned <= 0:
                need = quota_from_frac(mix, remaining)
            else:
                weights = {k: float(v) for k, v in need.items() if v > 0} or {
                    k: float(v) for k, v in mix.items()
                }
                need = quota_from_frac(weights, remaining)
        synth = [r for r in rows if not is_real_row(r)]
        synth_mix = {k: (need[k] / remaining) for k in need}
        tagged_synth = select_grpo_rows(
            synth,
            remaining,
            synth_mix,
            rng,
            stage=stage,
            empty_pools=empty_pools,
            empty_chrome_frac=empty_chrome_frac,
            keep_all_real=False,
        )
        picked = tagged_reals + tagged_synth
        rng.shuffle(picked)
        return picked[:n]

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene = grpo_scene(row)
        if scene:
            by_scene[scene].append(row)

    quotas = quota_from_frac(mix, n)
    picked: list[dict[str, Any]] = []
    leftover_need = 0
    for scene, q in quotas.items():
        candidates = by_scene.get(scene) or []
        chrome_frac = empty_chrome_frac if scene == "empty" else None
        if scene == "empty" and empty_pools:
            sub = defaultdict(list)
            for row in candidates:
                sub[str(_meta(row).get("pool_id") or "empty_clear")].append(row)
            take = min(q, sum(len(v) for v in sub.values()))
            sub_q = quota_from_frac(empty_pools, take)
            scene_rows: list[dict[str, Any]] = []
            for pool_id, pq in sub_q.items():
                scene_rows.extend(
                    pick_unique_page_first(sub.get(pool_id) or [], pq, rng, chrome_frac=chrome_frac)
                )
            if len(scene_rows) < take:
                used = {id(r) for r in scene_rows}
                rest = [r for r in candidates if id(r) not in used]
                scene_rows.extend(
                    pick_unique_page_first(rest, take - len(scene_rows), rng, chrome_frac=chrome_frac)
                )
        else:
            take = min(q, len(candidates))
            scene_rows = pick_unique_page_first(candidates, take, rng, chrome_frac=chrome_frac)
        leftover_need += q - len(scene_rows)
        picked.extend(_tag_row(r, scene=scene, stage=stage) for r in scene_rows)

    if leftover_need > 0:
        used = {str((_meta(r).get("sample_id") or id(r))) for r in picked}
        rest = [
            r
            for r in rows
            if grpo_scene(r) is not None and str((_meta(r).get("sample_id") or id(r))) not in used
        ]
        extra = pick_unique_page_first(rest, leftover_need, rng)
        for row in extra:
            scene = grpo_scene(row) or "empty"
            picked.append(_tag_row(row, scene=scene, stage=stage))

    return picked[:n]


def select_sliced_rows(
    rows: list[dict[str, Any]],
    n: int,
    mix: dict[str, float],
    rng: random.Random,
    *,
    slice_fn,
    stage: str,
    empty_chrome_frac: float | None = None,
) -> list[dict[str, Any]]:
    """Draw ``n`` rows by named slices (e.g. regular / multi_frag / semantic / empty)."""
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sl = slice_fn(row)
        if sl and sl in mix:
            by_slice[sl].append(row)
    quotas = quota_from_frac(mix, n)
    picked: list[dict[str, Any]] = []
    leftover_need = 0
    for scene, q in quotas.items():
        chrome_frac = empty_chrome_frac if scene == "empty" else None
        take = min(q, len(by_slice.get(scene) or []))
        chosen = pick_unique_page_first(by_slice.get(scene) or [], take, rng, chrome_frac=chrome_frac)
        leftover_need += q - len(chosen)
        picked.extend(_tag_row(r, scene=scene, stage=stage) for r in chosen)
    if leftover_need > 0:
        used = {str((_meta(r).get("sample_id") or id(r))) for r in picked}
        rest = [
            r
            for r in rows
            if slice_fn(r) in mix and str((_meta(r).get("sample_id") or id(r))) not in used
        ]
        extra = pick_unique_page_first(rest, leftover_need, rng)
        for row in extra:
            picked.append(_tag_row(row, scene=slice_fn(row) or "regular", stage=stage))
    return picked[:n]


def compose_grpo_split(
    *,
    src_dir: Path,
    out_dir: Path,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    seed = int(recipe.get("seed") or 42)
    stage = str(recipe.get("name") or "grpo_q1_hard")
    mix = {str(k): float(v) for k, v in (recipe.get("mix") or {}).items()}
    empty_pools = recipe.get("empty_pools")
    empty_pools = {str(k): float(v) for k, v in empty_pools.items()} if empty_pools else None
    empty_chrome_frac = recipe.get("empty_chrome_frac")
    empty_chrome_frac = float(empty_chrome_frac) if empty_chrome_frac is not None else None
    n_train = int(recipe["n_train"])
    n_val = int(recipe.get("n_val") or 0)

    train_src = load_jsonl(src_dir / "train.jsonl")
    rng = random.Random(seed)
    train = select_grpo_rows(
        train_src,
        n_train,
        mix,
        rng,
        stage=stage,
        empty_pools=empty_pools,
        empty_chrome_frac=empty_chrome_frac,
        keep_all_real=bool(recipe.get("keep_all_real")),
    )
    train = interleave_stage(train, rng)

    val: list[dict[str, Any]] = []
    val_mix_raw = recipe.get("val_mix")
    val_mix = {str(k): float(v) for k, v in val_mix_raw.items()} if val_mix_raw else mix
    if n_val > 0 and (src_dir / "val.jsonl").is_file():
        val_src = load_jsonl(src_dir / "val.jsonl")
        if val_mix_raw:
            val = select_sliced_rows(
                val_src,
                n_val,
                val_mix,
                random.Random(seed + 1),
                slice_fn=val_slice,
                stage=stage,
                empty_chrome_frac=empty_chrome_frac,
            )
        else:
            val = select_grpo_rows(
                val_src,
                n_val,
                mix,
                random.Random(seed + 1),
                stage=stage,
                empty_pools=empty_pools,
                empty_chrome_frac=empty_chrome_frac,
                keep_all_real=bool(recipe.get("keep_all_real")),
            )
        val = interleave_stage(val, random.Random(seed + 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": stage,
        "from": str(src_dir),
        "prompt_key": recipe.get("prompt_key") or "a2_v3",
        "n_train": write_rows(out_dir / "train.jsonl", train),
        "n_val": write_rows(out_dir / "val.jsonl", val) if val else 0,
        "mix": mix,
        "val_mix": val_mix,
        "empty_pools": empty_pools,
        "empty_chrome_frac": empty_chrome_frac,
        "keep_all_real": bool(recipe.get("keep_all_real")),
        "train": mix_summary(train),
        "val": mix_summary(val) if val else {},
        "train_by_scene": _scene_counts(train),
        "val_by_scene": _scene_counts(val) if val else {},
    }
    (out_dir / "split_meta.json").write_text(
        __import__("json").dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def _scene_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(_meta(row).get("grpo_scene") or grpo_scene(row) or "?")] += 1
    return dict(sorted(counts.items()))

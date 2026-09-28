"""Compose a stage split by sampling from already-built objective pools."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from point_ocr.dataset_format import load_jsonl
from point_ocr.pools.spec import quota_from_frac
from point_ocr.prompts import get_prompt


def _parse_stem_boost(raw: Any) -> dict[str, dict[str, Any]]:
    """``{pool_id: {stems: [...], frac: 0.7}}`` — frac of that pool's quota from those stems."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("stem_boost must be a mapping of pool_id → {stems, frac}")
    out: dict[str, dict[str, Any]] = {}
    for pool_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"stem_boost.{pool_id} must be a mapping")
        stems = [str(s).removesuffix(".html") for s in (spec.get("stems") or [])]
        frac = float(spec.get("frac") or 0.0)
        if not stems:
            raise ValueError(f"stem_boost.{pool_id} missing stems")
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"stem_boost.{pool_id}.frac must be in (0, 1], got {frac}")
        out[str(pool_id)] = {"stems": stems, "frac": frac}
    return out


def load_recipe(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"recipe {path} is not a mapping")
    mix = data.get("mix")
    if not isinstance(mix, dict) or not mix:
        raise ValueError(f"recipe {path} missing mix:")
    data["mix"] = {str(k): float(v) for k, v in mix.items()}
    data["n"] = int(data.get("n") or 15000)
    split = data.get("split") or {"train": 0.90, "val": 0.05, "test": 0.05}
    data["split"] = {str(k): float(v) for k, v in split.items()}
    data["name"] = str(data.get("name") or path.stem)
    data["stem_boost"] = _parse_stem_boost(data.get("stem_boost"))
    chrome = data.get("chrome")
    if chrome:
        if not isinstance(chrome, dict):
            raise ValueError(f"recipe {path} chrome: must be a mapping")
        none_f = float(chrome.get("none") or 0.0)
        chrome_f = float(chrome.get("chrome") or 0.0)
        if abs(none_f + chrome_f - 1.0) > 1e-6:
            raise ValueError(f"recipe {path} chrome none+chrome must sum to 1")
        data["chrome_frac"] = chrome_f
    else:
        data["chrome_frac"] = None
    return data


def _row_meta(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") or row.get("meta") or {}


def _rewrite_prompt(row: dict[str, Any], prompt_key: str) -> dict[str, Any]:
    prompt = get_prompt("POINT", prompt_key=prompt_key)
    msgs = list(row.get("messages") or [])
    if msgs and msgs[0].get("role") == "user":
        content = str(msgs[0].get("content") or "")
        if content.startswith("<image>"):
            msgs[0] = {**msgs[0], "content": "<image>" + prompt}
        else:
            msgs[0] = {**msgs[0], "content": prompt}
    meta = dict(_row_meta(row))
    meta["prompt_key"] = prompt_key
    out = dict(row)
    out["messages"] = msgs
    out["metadata"] = meta
    return out


def _row_stem(row: dict[str, Any]) -> str:
    return str(_row_meta(row).get("template_stem") or "")


def _chrome_bucket(row: dict[str, Any]) -> str:
    fam = str(_row_meta(row).get("chrome_family") or "none").strip().lower()
    return "none" if fam in {"", "none"} else "chrome"


def sample_pool_rows(
    rows: list[dict[str, Any]],
    q: int,
    rng: random.Random,
    boost: dict[str, Any] | None = None,
    *,
    pool_id: str = "",
    chrome_frac: float | None = None,
    require_occ: bool = True,
) -> list[dict[str, Any]]:
    """Take ``q`` rows without replacement. Optional stem_boost oversamples listed stems."""
    if require_occ:
        from point_ocr.pools.select import occupancy_in_train_range

        filtered = [
            r
            for r in rows
            if occupancy_in_train_range(_row_meta(r).get("text_occupancy"))
        ]
        rows = filtered
    if q <= 0:
        return []
    if len(rows) < q:
        raise SystemExit(
            f"pool {pool_id or '?'} has {len(rows)} samples, need {q}. "
            f"Rebuild with a larger --target."
        )
    if chrome_frac is not None:
        n_ch = quota_from_frac({"chrome": float(chrome_frac), "none": 1.0 - float(chrome_frac)}, q)
        by_ch: dict[str, list[dict[str, Any]]] = {"none": [], "chrome": []}
        for row in rows:
            by_ch[_chrome_bucket(row)].append(row)
        picked: list[dict[str, Any]] = []
        for key, need in (("none", n_ch["none"]), ("chrome", n_ch["chrome"])):
            bucket = by_ch[key]
            if len(bucket) < need:
                raise SystemExit(
                    f"pool {pool_id or '?'} chrome={key} has {len(bucket)}, need {need}. "
                    f"Rebuild with more {'unwrapped' if key == 'none' else 'chrome'} pages."
                )
            rng.shuffle(bucket)
            inner = sample_pool_rows(
                bucket, need, rng, boost, pool_id=f"{pool_id}:{key}", chrome_frac=None, require_occ=False
            )
            picked.extend(inner)
        rng.shuffle(picked)
        return picked

    if not boost:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:q]

    stems: list[str] = list(boost["stems"])
    frac = float(boost["frac"])
    n_boost = quota_from_frac({"boost": frac, "rest": max(1e-12, 1.0 - frac)}, q)["boost"]
    n_boost = min(n_boost, q)
    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stem[_row_stem(row)].append(row)
    for bucket in by_stem.values():
        rng.shuffle(bucket)

    stem_q = quota_from_frac({s: 1.0 / len(stems) for s in stems}, n_boost)
    picked: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    for s in stems:
        bucket = by_stem.get(s) or []
        take = int(stem_q.get(s) or 0)
        picked.extend(bucket[:take])
        leftover.extend(bucket[take:])
    rng.shuffle(leftover)
    if len(picked) < n_boost and leftover:
        extra = min(n_boost - len(picked), len(leftover))
        picked.extend(leftover[:extra])
        leftover = leftover[extra:]
    picked = picked[:n_boost]
    if len(picked) < n_boost:
        tag = pool_id or "?"
        print(
            f"[compose] stem_boost {tag}: wanted {n_boost} from {stems}, "
            f"got {len(picked)} (pool short; filling from other stems)",
            flush=True,
        )

    rest: list[dict[str, Any]] = []
    stem_set = set(stems)
    for stem, bucket in by_stem.items():
        if stem in stem_set:
            continue
        rest.extend(bucket)
    rng.shuffle(rest)
    n_rest = max(0, q - len(picked))
    if len(rest) < n_rest:
        raise SystemExit(
            f"pool {pool_id or '?'} stem_boost: need {n_rest} non-boost rows, "
            f"have {len(rest)}. Lower mix quota or stem_boost.frac."
        )
    picked.extend(rest[:n_rest])
    rng.shuffle(picked)
    return picked


def sample_from_pools(
    mix: dict[str, float],
    n: int,
    pools_root: Path,
    *,
    rng: random.Random,
    prompt_key: str | None = None,
    stem_boost: dict[str, dict[str, Any]] | None = None,
    chrome_frac: float | None = None,
) -> list[dict[str, Any]]:
    quotas = quota_from_frac(mix, n)
    boost_by_pool = stem_boost or {}
    picked: list[dict[str, Any]] = []
    for pool_id, q in quotas.items():
        jsonl = pools_root / pool_id / "point_sharegpt.jsonl"
        if not jsonl.is_file():
            raise FileNotFoundError(f"missing pool jsonl: {jsonl}")
        rows = load_jsonl(jsonl)
        chunk = sample_pool_rows(
            rows,
            q,
            rng,
            boost_by_pool.get(pool_id),
            pool_id=pool_id,
            chrome_frac=chrome_frac,
        )
        for row in chunk:
            meta = dict(_row_meta(row))
            meta["pool_id"] = pool_id
            row = dict(row)
            row["metadata"] = meta
            if prompt_key:
                row = _rewrite_prompt(row, prompt_key)
            picked.append(row)
    return picked


def split_counts(n: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    """Largest-remainder train/val/test counts. Keep a val and test row when n>=3."""
    test_frac = max(0.0, 1.0 - float(train_frac) - float(val_frac))
    d = quota_from_frac(
        {"train": float(train_frac), "val": float(val_frac), "test": test_frac},
        n,
    )
    n_train, n_val, n_test = int(d["train"]), int(d["val"]), int(d["test"])
    if n >= 3 and n_val == 0:
        n_val = 1
        n_train = max(0, n_train - 1)
    if n >= 3 and n_test == 0:
        n_test = 1
        n_train = max(0, n_train - 1)
    return n_train, n_val, n_test


def split_rows(
    rows: list[dict[str, Any]],
    *,
    train_frac: float,
    val_frac: float,
    rng: random.Random,
    stratify_key: str = "pool_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratify by pool so val/test keep the same mix as train."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(_row_meta(row).get(stratify_key) or "_none") + "|" + _chrome_bucket(row)].append(row)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_train, n_val, n_test = split_counts(len(bucket), train_frac, val_frac)
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val : n_train + n_val + n_test])
    return train, val, test


def _interleave_proportional(
    rows: list[dict[str, Any]],
    key_fn: Any,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Draw the next row from a pool with probability ∝ remaining count.

    Avoids a long tail of the largest pool after round-robin empties the rest.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row))].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    out: list[dict[str, Any]] = []
    while True:
        keys = [k for k, b in buckets.items() if b]
        if not keys:
            break
        weights = [len(buckets[k]) for k in keys]
        k = rng.choices(keys, weights=weights, k=1)[0]
        out.append(buckets[k].pop())
    return out


def interleave_stage(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Mix pools throughout the JSONL (train must not be concatenated by pool)."""

    def pool_key(row: dict[str, Any]) -> str:
        return str(_row_meta(row).get("pool_id") or "?")

    mixed = _interleave_proportional(rows, pool_key, rng)
    if len(mixed) <= 2:
        return mixed
    # Tiny local shuffle: break same-template pairs without unmixing pools.
    out = list(mixed)
    w = max(4, min(12, len(out) // 80 or 4))
    step = max(1, w // 2)
    for start in range(0, len(out), step):
        end = min(len(out), start + w)
        chunk = out[start:end]
        rng.shuffle(chunk)
        out[start:end] = chunk
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def mix_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pool_tmpl: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        meta = _row_meta(r)
        pool_tmpl[str(meta.get("pool_id"))][str(meta.get("template_stem"))] += 1
    return {
        "n": len(rows),
        "by_pool": dict(sorted(Counter(str(_row_meta(r).get("pool_id")) for r in rows).items())),
        "by_template": dict(
            sorted(Counter(str(_row_meta(r).get("template_stem")) for r in rows).items())
        ),
        "by_pool_template": {
            pid: dict(sorted(counts.items())) for pid, counts in sorted(pool_tmpl.items())
        },
        "by_chrome": dict(sorted(Counter(_chrome_bucket(r) for r in rows).items())),
        "n_neg": sum(
            1
            for r in rows
            if _row_meta(r).get("is_negative")
            or not str((r.get("messages") or [{}, {}])[-1].get("content") or "").strip()
        ),
    }

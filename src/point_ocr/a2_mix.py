"""A2 curriculum mix quotas and sample interleaving.

Quotas match ``docs/CURRICULUM_PLAN.md`` §2.2. Interleaving keeps templates
and slices from clustering in the written JSONL (Trainer also shuffles each
epoch, but on-disk order should already be mixed for inspection / non-shuffled
paths).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Hashable, Iterable, Sequence, TypeVar

T = TypeVar("T")

# Target total for full A2 build (16k–20k allowed; 18k default).
A2_TARGET_DEFAULT = 18_000

# Fraction of total samples per ``a2_slice`` (must sum ≈ 1.0).
A2_SLICE_FRAC: dict[str, float] = {
    "replay": 0.22,
    "adjacency": 0.14,
    "multi_frag": 0.10,
    "semantic_group": 0.14,
    "corner_extreme": 0.06,
    "special": 0.12,  # table HTML + formula/code/image deny + prose controls
    "desktop": 0.14,
    "near_edge": 0.08,
}

# Templates preferred per slice (round-robin within the slice quota).
A2_SLICE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "replay": (
        "01_article_twocol.html",
        "08_github_readme_dark.html",
        "12_wikipedia_article.html",
        "20_forum_zh.html",
        "24_docs_portal_dense.html",
    ),
    "adjacency": (
        "01_article_twocol.html",
        "04_zh_news_twocol.html",
        "14_magazine_3col.html",
    ),
    "multi_frag": (
        "01_article_twocol.html",
        "14_magazine_3col.html",
        "04_zh_news_twocol.html",
    ),
    "semantic_group": ("27_a2_semantic_groups.html",),
    "corner_extreme": (
        "01_article_twocol.html",
        "12_wikipedia_article.html",
        "20_forum_zh.html",
        "24_docs_portal_dense.html",
    ),
    "special": (
        "28_a2_special_semantics.html",
        "07_formulas_defs.html",
        "03_code_ui.html",
        "06_dense_table_cells.html",
    ),
    "desktop": (
        "21_desktop_stacked_windows.html",
        "22_desktop_zh_messy.html",
        "23_ide_dense_split.html",
        "25_desktop_win11_collage.html",
        "26_desktop_mac_collage.html",
    ),
    "near_edge": (
        "01_article_twocol.html",
        "21_desktop_stacked_windows.html",
        "25_desktop_win11_collage.html",
    ),
}


def quota_counts(n_total: int = A2_TARGET_DEFAULT) -> dict[str, int]:
    """Integer quotas that sum exactly to ``n_total`` (largest-remainder)."""
    keys = list(A2_SLICE_FRAC.keys())
    raw = [A2_SLICE_FRAC[k] * n_total for k in keys]
    base = [int(x) for x in raw]
    rem = n_total - sum(base)
    order = sorted(range(len(keys)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return dict(zip(keys, base))


def template_stem_from_page_id(page_id: str) -> str:
    """Best-effort stem from page_id like ``desk__23_ide_dense_split`` / ``replay__01_…__0``."""
    if not page_id:
        return "unknown"
    parts = page_id.split("__")
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2:
        return parts[1]
    return parts[0]


def interleave_by_key(
    items: Sequence[T],
    key_fn: Callable[[T], Hashable],
    rng: random.Random,
) -> list[T]:
    """Shuffle within each key bucket, then round-robin across keys.

    Guarantees consecutive items rarely share the same key when buckets are
    similarly sized; with uneven buckets, leftovers are appended shuffled.
    """
    buckets: dict[Hashable, list[T]] = defaultdict(list)
    for it in items:
        buckets[key_fn(it)].append(it)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(buckets[k])

    out: list[T] = []
    while True:
        progress = False
        rng.shuffle(keys)
        for k in keys:
            bucket = buckets[k]
            if bucket:
                out.append(bucket.pop())
                progress = True
        if not progress:
            break
    return out


def interleave_a2_samples(
    samples: Sequence[T],
    *,
    rng: random.Random,
    slice_fn: Callable[[T], str],
    template_fn: Callable[[T], str],
) -> list[T]:
    """Two-level mix: round-robin by ``(slice, template)``, then light global shuffle.

    The final shuffle uses a Fisher–Yates with a *local window* so order stays
    mixed without rebuilding long same-template runs.
    """
    mixed = interleave_by_key(
        samples,
        key_fn=lambda s: (slice_fn(s), template_fn(s)),
        rng=rng,
    )
    return _window_shuffle(mixed, rng, window=max(8, len(mixed) // 50 or 8))


def _window_shuffle(items: list[T], rng: random.Random, *, window: int) -> list[T]:
    """Shuffle inside overlapping windows to break residual local clumps."""
    if len(items) <= 2:
        return list(items)
    out = list(items)
    w = max(2, min(window, len(out)))
    step = max(1, w // 2)
    for start in range(0, len(out), step):
        end = min(len(out), start + w)
        chunk = out[start:end]
        rng.shuffle(chunk)
        out[start:end] = chunk
    return out


def meta_template_stem(meta: dict[str, Any] | None) -> str:
    m = meta or {}
    if m.get("template_stem"):
        return str(m["template_stem"])
    if m.get("template"):
        t = str(m["template"])
        return t[:-5] if t.endswith(".html") else t
    return template_stem_from_page_id(str(m.get("page_id") or ""))


def summarize_mix(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Quick stats for written ShareGPT rows (metadata nested)."""
    from collections import Counter

    slice_c: Counter[str] = Counter()
    tmpl_c: Counter[str] = Counter()
    max_run = 0
    cur_key = None
    run = 0
    n = 0
    for row in rows:
        meta = row.get("metadata") or {}
        sl = str(meta.get("a2_slice") or "?")
        tm = meta_template_stem(meta)
        slice_c[sl] += 1
        tmpl_c[tm] += 1
        key = (sl, tm)
        if key == cur_key:
            run += 1
        else:
            max_run = max(max_run, run)
            cur_key = key
            run = 1
        n += 1
    max_run = max(max_run, run)
    return {
        "n": n,
        "by_slice": dict(sorted(slice_c.items())),
        "by_template": dict(sorted(tmpl_c.items())),
        "max_consecutive_same_slice_template": max_run,
    }

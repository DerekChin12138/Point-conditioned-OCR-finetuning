"""A2_v2 curriculum mix quotas — hard cases + A1 replay.

Absorbs corners / adjacency / multi-frag / short-group that A1 no longer covers.
"""

from __future__ import annotations

A2_V2_TARGET_DEFAULT = 16_000

A2_V2_SLICE_FRAC: dict[str, float] = {
    "replay_long": 0.24,
    "adjacency": 0.16,
    "multi_frag": 0.16,
    "semantic_group": 0.20,
    "corner_extreme": 0.12,
    "empty_neg": 0.10,
    "special_light": 0.02,
}

A2_V2_SLICE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "replay_long": (
        "01_article_twocol.html",
        "04_zh_news_twocol.html",
        "08_github_readme_dark.html",
        "12_wikipedia_article.html",
        "14_magazine_3col.html",
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
    "semantic_group": (
        "27_a2_semantic_groups.html",
        "29_a2_semantic_dark_docs.html",
        "30_a2_semantic_news_cards.html",
        "31_a2_semantic_wiki_twocol.html",
        "32_a2_semantic_release_notes.html",
    ),
    "corner_extreme": (
        "12_wikipedia_article.html",
        "20_forum_zh.html",
        "24_docs_portal_dense.html",
        "08_github_readme_dark.html",
    ),
    "empty_neg": (
        "01_article_twocol.html",
        "08_github_readme_dark.html",
        "12_wikipedia_article.html",
        "20_forum_zh.html",
    ),
    "special_light": (
        "28_a2_special_semantics.html",
        "07_formulas_defs.html",
        "03_code_ui.html",
        "06_dense_table_cells.html",
    ),
}

# Static HTML (no content-pool fill)
A2_V2_STATIC_SLICES = frozenset({"semantic_group", "special_light"})


def a2_v2_quota_counts(n_total: int = A2_V2_TARGET_DEFAULT) -> dict[str, int]:
    keys = list(A2_V2_SLICE_FRAC.keys())
    raw = [A2_V2_SLICE_FRAC[k] * n_total for k in keys]
    base = [int(x) for x in raw]
    rem = n_total - sum(base)
    order = sorted(range(len(keys)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return dict(zip(keys, base))

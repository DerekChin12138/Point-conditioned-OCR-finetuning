"""A1_v2 curriculum mix quotas — easy foundation only.

Hard cases (corners, adjacency / short-block context, multi-frag) live in A2.
"""

from __future__ import annotations

A1_V2_TARGET_DEFAULT = 15_000

# long paragraphs + empty negatives only
A1_V2_SLICE_FRAC: dict[str, float] = {
    "long_center": 0.82,
    "empty_neg": 0.18,
}

A1_V2_SLICE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "long_center": (
        "01_article_twocol.html",
        "04_zh_news_twocol.html",
        "08_github_readme_dark.html",
        "12_wikipedia_article.html",
        "14_magazine_3col.html",
        "20_forum_zh.html",
        "24_docs_portal_dense.html",
    ),
    "empty_neg": (
        "01_article_twocol.html",
        "08_github_readme_dark.html",
        "12_wikipedia_article.html",
        "20_forum_zh.html",
    ),
}


def a1_v2_quota_counts(n_total: int = A1_V2_TARGET_DEFAULT) -> dict[str, int]:
    keys = list(A1_V2_SLICE_FRAC.keys())
    raw = [A1_V2_SLICE_FRAC[k] * n_total for k in keys]
    base = [int(x) for x in raw]
    rem = n_total - sum(base)
    order = sorted(range(len(keys)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return dict(zip(keys, base))

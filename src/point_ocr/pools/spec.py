"""Registry of objective sample pools (not training stages)."""

from __future__ import annotations

from dataclasses import dataclass, field


DOC_TEMPLATES: tuple[str, ...] = (
    "01_article_twocol.html",
    "04_zh_news_twocol.html",
    "08_github_readme_dark.html",
    "12_wikipedia_article.html",
    "14_magazine_3col.html",
    "20_forum_zh.html",
    "24_docs_portal_dense.html",
    "05_pdf_academic.html",
    "10_ecommerce_zh.html",
    "11_email_client.html",
    "13_settings_form.html",
    "15_slides_dark.html",
    "16_invoice.html",
    "18_kanban.html",
    "33_q1_reader_airy.html",
    "34_q1_reader_mid.html",
    "35_q1_reader_packed.html",
    "36_q1_twocol_fill.html",
)

ADJACENT_TEMPLATES: tuple[str, ...] = (
    "01_article_twocol.html",
    "04_zh_news_twocol.html",
    "14_magazine_3col.html",
)

MULTI_FRAG_TEMPLATES: tuple[str, ...] = (
    "01_article_twocol.html",
    "14_magazine_3col.html",
    "04_zh_news_twocol.html",
)

SEMANTIC_TEMPLATES: tuple[str, ...] = (
    "27_a2_semantic_groups.html",
    "29_a2_semantic_dark_docs.html",
    "30_a2_semantic_news_cards.html",
    "31_a2_semantic_wiki_twocol.html",
    "32_a2_semantic_release_notes.html",
)

SPECIAL_CONTEXT_TEMPLATES: tuple[str, ...] = (
    "02_table_formula.html",
    "03_code_ui.html",
    "06_dense_table_cells.html",
    "07_formulas_defs.html",
    "28_a2_special_semantics.html",
)

# Pages that must not be filled from the LLM content pool (layout is authored).
STATIC_HTML_FILES: frozenset[str] = frozenset((*SEMANTIC_TEMPLATES, *SPECIAL_CONTEXT_TEMPLATES))

# Visual diversity: document pages plus table/image scenery (formula/code are text).
# Markers on tables/images → empty; see select.is_special_block.
CORE_TEMPLATES: tuple[str, ...] = DOC_TEMPLATES + SPECIAL_CONTEXT_TEMPLATES
CLEAR_TEMPLATES: tuple[str, ...] = DOC_TEMPLATES[:8] + SPECIAL_CONTEXT_TEMPLATES
BOUNDARY_TEMPLATES: tuple[str, ...] = ADJACENT_TEMPLATES + ("06_dense_table_cells.html",)
SPECIAL_EMPTY_TEMPLATES: tuple[str, ...] = (
    "02_table_formula.html",
    "06_dense_table_cells.html",
    "28_a2_special_semantics.html",
)


@dataclass(frozen=True)
class PoolSpec:
    pool_id: str
    goal: str
    gt: str
    templates: tuple[str, ...]
    static_html: bool = False
    default_target: int = 8_000
    default_prompt_key: str = "a2_v2"
    extras: dict = field(default_factory=dict)


POOL_SPECS: dict[str, PoolSpec] = {
    "core_inner": PoolSpec(
        pool_id="core_inner",
        goal=(
            "Marker center in the inner 80% area of a long in-frame text block. "
            "Basic POINT: return only that block."
        ),
        gt="that block's Markdown",
        templates=CORE_TEMPLATES,
        default_target=15_000,
        default_prompt_key="a2_v2",
    ),
    "empty_clear": PoolSpec(
        pool_id="empty_clear",
        goal="Marker not on any text (clear of ink + clearance). Return empty.",
        gt="empty string",
        templates=CLEAR_TEMPLATES,
        default_target=7_500,
        default_prompt_key="a2_v2",
    ),
    "empty_special": PoolSpec(
        pool_id="empty_special",
        goal=(
            "Marker center inside a table or image/chart bbox. "
            "Return empty. Formulas and code are normal text, not this pool."
        ),
        gt="empty string",
        templates=SPECIAL_EMPTY_TEMPLATES,
        default_target=4_000,
        default_prompt_key="a2_v3",
    ),
    "empty_boundary": PoolSpec(
        pool_id="empty_boundary",
        goal=(
            "Marker in the junction of two nearby text *units* (a related short "
            "cluster is one bbox), strictly outside every unit's 100% bbox. "
            "The inner-80%–100% ring is not sampled. Intra-cluster gutters are hits."
        ),
        gt="empty string",
        templates=BOUNDARY_TEMPLATES,
        default_target=5_000,
        default_prompt_key="a2_v2",
    ),
    "multi_frag": PoolSpec(
        pool_id="multi_frag",
        goal=(
            "Wrapped / multi-column paragraph. Marker in any fragment bbox "
            "→ return the full block Markdown."
        ),
        gt="full block Markdown (all fragments)",
        templates=MULTI_FRAG_TEMPLATES,
        default_target=5_000,
        default_prompt_key="a2_v2",
        extras={"min_css_pixels": 1024 * 720, "min_aspect": 1.0},
    ),
    "semantic_group": PoolSpec(
        pool_id="semantic_group",
        goal=(
            "Tightly related short blocks as one bbox (seed+body). Marker on a "
            "member *or in the gutter between members* → whole group, no extra "
            "neighbors."
        ),
        gt="group Markdown (positives); own Markdown for truncate-neighbors",
        templates=SEMANTIC_TEMPLATES,
        static_html=True,
        default_target=5_000,
        default_prompt_key="a2_v2",
    ),
}


def list_pool_ids() -> list[str]:
    return list(POOL_SPECS.keys())


def get_pool_spec(pool_id: str) -> PoolSpec:
    try:
        return POOL_SPECS[pool_id]
    except KeyError as e:
        raise KeyError(f"unknown pool {pool_id!r}; expected one of {list_pool_ids()}") from e


def quota_from_frac(frac: dict[str, float], n_total: int) -> dict[str, int]:
    """Largest-remainder integer quotas that sum to ``n_total``."""
    keys = list(frac.keys())
    raw = [float(frac[k]) * n_total for k in keys]
    base = [int(x) for x in raw]
    rem = n_total - sum(base)
    order = sorted(range(len(keys)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in order[: max(0, rem)]:
        base[i] += 1
    return dict(zip(keys, base))

"""Objective-pool registry, emit, and stage compose."""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from point_ocr.build_point import BlockAnno
from point_ocr.pools.compose import (
    interleave_stage,
    load_recipe,
    sample_from_pools,
    sample_pool_rows,
    split_counts,
    split_rows,
)
from point_ocr.pools.emit import emit_pool_samples
from point_ocr.pools.spec import POOL_SPECS, quota_from_frac
from point_ocr.sample_points import BBox


def test_quota_from_frac_sums():
    q = quota_from_frac({"a": 0.82, "b": 0.18}, 15000)
    assert sum(q.values()) == 15000
    assert q["a"] == 12300
    assert q["b"] == 2700


def test_recipes_parse(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    a1 = load_recipe(root / "data/recipes/stage_a1.yaml")
    a2 = load_recipe(root / "data/recipes/stage_a2.yaml")
    assert abs(sum(a1["mix"].values()) - 1.0) < 1e-6
    assert abs(sum(a2["mix"].values()) - 1.0) < 1e-6
    assert a1["n"] == 20000
    assert a2["n"] == 15000
    assert a1["mix"]["core_inner"] == 0.60
    assert a2["mix"]["multi_frag"] == 0.15
    from point_ocr.pools.spec import quota_from_frac

    q1 = quota_from_frac(a1["mix"], a1["n"])
    q2 = quota_from_frac(a2["mix"], a2["n"])
    assert sum(q1.values()) == 20000
    assert q1 == {
        "core_inner": 12000,
        "empty_clear": 5000,
        "empty_boundary": 1000,
        "multi_frag": 1000,
        "semantic_group": 1000,
    }
    assert sum(q2.values()) == 15000
    assert q2 == {
        "core_inner": 6000,
        "empty_clear": 2250,
        "empty_boundary": 2250,
        "multi_frag": 2250,
        "semantic_group": 2250,
    }
    a2b = load_recipe(root / "data/recipes/stage_a2b.yaml")
    b1 = load_recipe(root / "data/recipes/stage_b1.yaml")
    assert abs(sum(a2b["mix"].values()) - 1.0) < 1e-6
    assert abs(sum(b1["mix"].values()) - 1.0) < 1e-6
    assert a2b["n"] == 16000
    assert b1["n"] == 25000
    assert b1["chrome_frac"] == 0.40
    assert b1["mix"]["empty_clear"] == 0.20
    q2b = quota_from_frac(a2b["mix"], a2b["n"])
    assert sum(q2b.values()) == 16000
    assert q2b == {
        "core_inner": 3520,
        "empty_clear": 1280,
        "empty_boundary": 3840,
        "multi_frag": 3840,
        "semantic_group": 3520,
    }
    assert a1["prompt_key"] == "a2_v2"
    assert a2["prompt_key"] == "a2_v2"
    assert a2b["prompt_key"] == "a2_v2"
    assert b1["prompt_key"] == "a2_v2"
    q1 = load_recipe(root / "data/recipes/stage_q1.yaml")
    assert abs(sum(q1["mix"].values()) - 1.0) < 1e-6
    assert q1["n"] == 25000
    assert q1["prompt_key"] == "a2_v3"
    assert q1["chrome_frac"] == 0.40
    assert "empty_boundary" not in q1["mix"]
    assert q1["mix"]["empty_special"] == 0.10
    assert q1["mix"]["empty_clear"] == 0.15
    assert set(a1["mix"]) <= set(POOL_SPECS)
    assert set(a2["mix"]) <= set(POOL_SPECS)
    assert set(a2b["mix"]) <= set(POOL_SPECS)
    assert set(b1["mix"]) <= set(POOL_SPECS)
    assert set(q1["mix"]) <= set(POOL_SPECS)
    qb1 = quota_from_frac(b1["mix"], b1["n"])
    assert sum(qb1.values()) == 25000
    assert qb1 == {
        "core_inner": 15000,
        "empty_clear": 5000,
        "empty_boundary": 1250,
        "multi_frag": 1875,
        "semantic_group": 1875,
    }
    qq1 = quota_from_frac(q1["mix"], q1["n"])
    assert sum(qq1.values()) == 25000
    assert qq1 == {
        "core_inner": 15000,
        "empty_clear": 3750,
        "empty_special": 2500,
        "multi_frag": 1875,
        "semantic_group": 1875,
    }
    assert a2b["stem_boost"]["core_inner"]["frac"] == 0.70
    assert "01_article_twocol" in a2b["stem_boost"]["core_inner"]["stems"]
    assert a2b["stem_boost"]["semantic_group"]["stems"] == ["30_a2_semantic_news_cards"]


def test_compose_samples_and_splits(tmp_path: Path):
    pools = tmp_path / "pools"
    for pid, n in (("core_inner", 20), ("empty_clear", 10)):
        d = pools / pid
        d.mkdir(parents=True)
        with (d / "point_sharegpt.jsonl").open("w", encoding="utf-8") as f:
            for i in range(n):
                rec = {
                    "messages": [
                        {"role": "user", "content": "<image>PROMPT"},
                        {"role": "assistant", "content": "text" if pid == "core_inner" else ""},
                    ],
                    "images": [str(d / f"{i}.jpg")],
                    "metadata": {
                        "sample_id": f"{pid}:{i}",
                        "pool_id": pid,
                        "template_stem": "01_article_twocol",
                        "is_negative": pid == "empty_clear",
                        "prompt_key": "a1_v2",
                    },
                }
                f.write(json.dumps(rec) + "\n")
    rng = random.Random(0)
    rows = sample_from_pools(
        {"core_inner": 0.8, "empty_clear": 0.2},
        10,
        pools,
        rng=rng,
        prompt_key="a1_v2",
    )
    assert len(rows) == 10
    assert sum(1 for r in rows if r["metadata"]["pool_id"] == "core_inner") == 8
    train, val, test = split_rows(rows, train_frac=0.6, val_frac=0.2, rng=random.Random(1))
    assert len(train) + len(val) + len(test) == 10
    assert val and test


def test_stem_boost_takes_listed_stems_first():
    rows = []
    for stem, n in (("01_article_twocol", 20), ("14_magazine_3col", 20), ("20_forum_zh", 30)):
        for i in range(n):
            rows.append({"metadata": {"template_stem": stem, "sample_id": f"{stem}:{i}"}})
    picked = sample_pool_rows(
        rows,
        20,
        random.Random(0),
        {"stems": ["01_article_twocol", "14_magazine_3col"], "frac": 0.70},
        pool_id="core_inner",
    )
    assert len(picked) == 20
    stems = [r["metadata"]["template_stem"] for r in picked]
    n01 = stems.count("01_article_twocol")
    n14 = stems.count("14_magazine_3col")
    n20 = stems.count("20_forum_zh")
    assert n01 + n14 == 14
    assert abs(n01 - n14) <= 1
    assert n20 == 6
    ids = [r["metadata"]["sample_id"] for r in picked]
    assert len(ids) == len(set(ids))


def test_stem_boost_caps_when_stem_pool_is_short():
    rows = []
    for stem, n in (("30_a2_semantic_news_cards", 4), ("27_a2_semantic_groups", 20)):
        for i in range(n):
            rows.append({"metadata": {"template_stem": stem, "sample_id": f"{stem}:{i}"}})
    picked = sample_pool_rows(
        rows,
        10,
        random.Random(1),
        {"stems": ["30_a2_semantic_news_cards"], "frac": 0.50},
        pool_id="semantic_group",
    )
    stems = [r["metadata"]["template_stem"] for r in picked]
    assert stems.count("30_a2_semantic_news_cards") == 4
    assert stems.count("27_a2_semantic_groups") == 6


def test_split_counts_keeps_val_test():
    assert split_counts(15000, 0.90, 0.05) == (13500, 750, 750)
    n_train, n_val, n_test = split_counts(750, 0.90, 0.05)
    assert n_train + n_val + n_test == 750
    assert n_val >= 1 and n_test >= 1


def test_interleave_stage_mixes_pools():
    rows = []
    for pid, n in (("core_inner", 80), ("empty_clear", 20), ("multi_frag", 20)):
        for i in range(n):
            rows.append({"metadata": {"pool_id": pid, "template_stem": "t", "sample_id": f"{pid}:{i}"}})
    mixed = interleave_stage(rows, random.Random(0))
    assert [r["metadata"]["sample_id"] for r in mixed] != [r["metadata"]["sample_id"] for r in rows]
    prefix = [r["metadata"]["pool_id"] for r in mixed[:30]]
    assert len(set(prefix)) >= 2
    run = max_run = 1
    prev = mixed[0]["metadata"]["pool_id"]
    for r in mixed[1:]:
        cur = r["metadata"]["pool_id"]
        if cur == prev:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
            prev = cur
    assert max_run < 25


def test_emit_core_inner_and_boundary(tmp_path: Path):
    img = Image.new("RGB", (800, 600), (245, 245, 240))
    left = BlockAnno(
        "p1",
        BBox(80, 80, 280, 420),
        markdown="The river valley survey notes a slow silt load after last week's rain.",
        rects=[BBox(80, 80, 280, 420)],
    )
    right = BlockAnno(
        "p2",
        BBox(310, 80, 510, 420),
        markdown="Harbor logs list three inbound barges and a delayed customs inspection.",
        rects=[BBox(310, 80, 510, 420)],
    )
    marked = tmp_path / "marked"
    core = emit_pool_samples(
        img,
        [left, right],
        pool_id="core_inner",
        page_id="t_core",
        marked_dir=marked,
        seed=1,
        template_stem="01_article_twocol",
        prompt_key="a1_v2",
    )
    assert core
    assert all(s.target.strip() for s in core)
    regions = {s.meta.get("region") for s in core}
    assert regions == {"diamond", "nw", "ne", "sw", "se"}
    by_block: dict[str, int] = {}
    for s in core:
        by_block[str(s.meta.get("block_id"))] = by_block.get(str(s.meta.get("block_id")), 0) + 1
    assert all(n == 10 for n in by_block.values())

    bound = emit_pool_samples(
        img,
        [left, right],
        pool_id="empty_boundary",
        page_id="t_bound",
        marked_dir=marked,
        seed=2,
        template_stem="01_article_twocol",
        prompt_key="a1_v2",
    )
    assert bound
    assert all(s.target == "" for s in bound)
    assert all(s.meta.get("is_negative") for s in bound)
    assert all(s.meta.get("region") == "neg_boundary" for s in bound)
    for s in bound:
        x, y = s.meta["point"]
        assert not left.bbox.contains(x, y)
        assert not right.bbox.contains(x, y)


def test_short_cluster_is_one_bbox_not_boundary_empty(tmp_path: Path):
    from point_ocr.pools.units import clustered_member_ids, occupancy_boxes
    from point_ocr.sample_points import sample_boundary_empty_points

    heading = BlockAnno(
        "h2b",
        BBox(40, 40, 220, 64),
        markdown="Checklist",
        extra={"tag": "h2"},
        rects=[BBox(40, 40, 220, 64)],
    )
    li1 = BlockAnno(
        "li1",
        BBox(40, 72, 320, 94),
        markdown="Clone the repository",
        extra={"tag": "li"},
        rects=[BBox(40, 72, 320, 94)],
    )
    li2 = BlockAnno(
        "li2",
        BBox(40, 102, 320, 124),
        markdown="Install deps with uv sync",
        extra={"tag": "li"},
        rects=[BBox(40, 102, 320, 124)],
    )
    long = BlockAnno(
        "p1",
        BBox(40, 148, 420, 520),
        markdown="The river valley survey notes a slow silt load after last week's rain and a delayed barge.",
        extra={"tag": "p"},
        rects=[BBox(40, 148, 420, 520)],
    )
    blocks = [heading, li1, li2, long]
    ids = clustered_member_ids(blocks, page_html="")
    assert ids == {"h2b", "li1", "li2"}
    occ = occupancy_boxes(blocks, page_html="")
    assert len(occ) == 2

    rng = random.Random(0)
    pts = sample_boundary_empty_points(800, 600, occ, n=20, rng=rng)
    assert pts
    for p in pts:
        assert all(not b.contains(p.x, p.y) for b in occ)

    img = Image.new("RGB", (800, 600), (245, 245, 240))
    bound = emit_pool_samples(
        img,
        blocks,
        pool_id="empty_boundary",
        page_id="t_list",
        marked_dir=tmp_path / "marked_list",
        seed=3,
        template_stem="01_article_twocol",
        prompt_key="a2_v2",
    )
    for s in bound:
        x, y = s.meta["point"]
        assert not (40 <= x <= 320 and li1.bbox.y1 <= y <= li2.bbox.y0), (x, y)


def test_semantic_group_union_includes_member_gutter(tmp_path: Path):
    html = """
    <html><body>
    <section data-semantic-group="g_hl" data-group-kind="heading_list">
      <h2 data-block-id="seed" data-role="seed">安装步骤</h2>
      <ul>
        <li data-block-id="li1" data-role="body">克隆仓库</li>
        <li data-block-id="li2" data-role="body">安装依赖</li>
      </ul>
    </section>
    </body></html>
    """
    seed = BlockAnno(
        "seed",
        BBox(40, 40, 200, 64),
        markdown="## 安装步骤",
        extra={"tag": "h2", "semantic_group": "g_hl"},
        rects=[BBox(40, 40, 200, 64)],
    )
    li1 = BlockAnno(
        "li1",
        BBox(40, 80, 280, 102),
        markdown="克隆仓库",
        extra={"tag": "li", "semantic_group": "g_hl"},
        rects=[BBox(40, 80, 280, 102)],
    )
    li2 = BlockAnno(
        "li2",
        BBox(40, 118, 280, 140),
        markdown="安装依赖",
        extra={"tag": "li", "semantic_group": "g_hl"},
        rects=[BBox(40, 118, 280, 140)],
    )
    img = Image.new("RGB", (800, 600), (245, 245, 240))
    samples = emit_pool_samples(
        img,
        [seed, li1, li2],
        pool_id="semantic_group",
        page_id="t_sg",
        marked_dir=tmp_path / "marked_sg",
        seed=4,
        template_stem="27_a2_semantic_groups",
        prompt_key="a2_v2",
        page_html=html,
    )
    pos = [s for s in samples if s.meta.get("a2_kind") == "semantic_group"]
    assert pos
    assert all("安装步骤" in s.target and "克隆仓库" in s.target for s in pos)
    assert len(pos) == 10
    assert {s.meta.get("region") for s in pos} == {"diamond", "nw", "ne", "sw", "se"}


def test_special_blocks_table_image_only_not_formula():
    from point_ocr.pools.select import is_special_block, large_blocks

    table = BlockAnno(
        "tbl",
        BBox(80, 80, 400, 300),
        markdown="<table><tr><td>alpha beta gamma delta epsilon</td></tr></table>",
        extra={"tag": "table", "a2_kind": "table"},
        rects=[BBox(80, 80, 400, 300)],
    )
    formula = BlockAnno(
        "eq1",
        BBox(80, 80, 220, 140),
        markdown="L = -\\sum_t \\log p",
        extra={"tag": "div", "a2_kind": "formula", "data-latex": "L"},
        rects=[BBox(80, 80, 220, 140)],
    )
    prose_dollar = BlockAnno(
        "p_dollar",
        BBox(80, 320, 400, 520),
        markdown="Cost is about $12 per unit after the \\sum discount.",
        extra={"tag": "p"},
        rects=[BBox(80, 320, 400, 520)],
    )
    prose = BlockAnno(
        "p1",
        BBox(80, 320, 400, 520),
        markdown="The river valley survey notes a slow silt load after last week's rain.",
        extra={"tag": "p"},
        rects=[BBox(80, 320, 400, 520)],
    )
    assert is_special_block(table)
    assert not is_special_block(formula)
    assert not is_special_block(prose_dollar)
    assert not is_special_block(prose)
    kept = large_blocks([table, formula, prose], 800 * 600, image_w=800, image_h=600)
    assert [b.block_id for b in kept] == ["p1"] or "eq1" in {b.block_id for b in kept}
    assert "tbl" not in {b.block_id for b in kept}


def test_text_occupancy_frac_union_not_sum():
    from point_ocr.pools.select import text_occupancy_frac

    a = BlockAnno("a", BBox(0, 0, 100, 100), markdown="aaaa", extra={"tag": "p"}, rects=[BBox(0, 0, 100, 100)])
    b = BlockAnno("b", BBox(50, 0, 150, 100), markdown="bbbb", extra={"tag": "p"}, rects=[BBox(50, 0, 150, 100)])
    occ = text_occupancy_frac([a, b], 200, 100, scale=1)
    # Union width 150 / 200 = 0.75, not sum 1.0.
    assert 0.70 <= occ <= 0.80


def test_q1_reader_templates_registered():
    from point_ocr.pools.spec import CORE_TEMPLATES

    for name in (
        "33_q1_reader_airy.html",
        "34_q1_reader_mid.html",
        "35_q1_reader_packed.html",
        "36_q1_twocol_fill.html",
    ):
        assert name in CORE_TEMPLATES


def test_empty_special_points_land_on_table_or_image_not_formula(tmp_path: Path):
    formula = BlockAnno(
        "eq1",
        BBox(40, 40, 220, 120),
        markdown="L = -\\sum_t \\log p",
        extra={"tag": "div", "a2_kind": "formula", "data-latex": "L"},
        rects=[BBox(40, 40, 220, 120)],
    )
    table = BlockAnno(
        "tbl",
        BBox(260, 40, 520, 220),
        markdown="<table><tr><td>alpha beta</td></tr></table>",
        extra={"tag": "table", "a2_kind": "table"},
        rects=[BBox(260, 40, 520, 220)],
    )
    image = BlockAnno(
        "img1",
        BBox(40, 240, 220, 400),
        markdown="",
        extra={"tag": "img", "a2_kind": "image"},
        rects=[BBox(40, 240, 220, 400)],
    )
    prose = BlockAnno(
        "p1",
        BBox(260, 280, 520, 520),
        markdown="The river valley survey notes a slow silt load after last week's rain.",
        extra={"tag": "p"},
        rects=[BBox(260, 280, 520, 520)],
    )
    img = Image.new("RGB", (800, 600), (245, 245, 240))
    samples = emit_pool_samples(
        img,
        [formula, table, prose, image],
        pool_id="empty_special",
        page_id="t_spec",
        marked_dir=tmp_path / "marked_special",
        seed=7,
        template_stem="28_a2_special_semantics",
        prompt_key="a2_v3",
    )
    assert samples
    assert all(s.target == "" for s in samples)
    assert all(s.meta.get("is_negative") for s in samples)
    hits = {s.meta.get("special_block_id") for s in samples}
    assert hits <= {"tbl", "img1"}
    assert "eq1" not in hits
    for s in samples:
        x, y = s.meta["point"]
        on_table = table.bbox.contains(x, y)
        on_image = image.bbox.contains(x, y)
        assert on_table or on_image
        assert not formula.bbox.contains(x, y)
        assert not prose.bbox.contains(x, y)
        assert str(s.meta.get("region") or "").startswith("special_")

def test_empty_clear_uses_wide_rect_gap(tmp_path: Path):
    from point_ocr.sample_points import Q1_NEG_CLEARANCE_PX

    prose = BlockAnno(
        "p1",
        BBox(80, 80, 400, 360),
        markdown="The river valley survey notes a slow silt load after last week's rain and a delayed barge.",
        extra={"tag": "p"},
        rects=[BBox(80, 80, 400, 360)],
    )
    img = Image.new("RGB", (800, 600), (245, 245, 240))
    samples = emit_pool_samples(
        img,
        [prose],
        pool_id="empty_clear",
        page_id="t_clear",
        marked_dir=tmp_path / "marked_clear",
        seed=11,
        template_stem="01_article_twocol",
        prompt_key="a2_v3",
    )
    negs = [s for s in samples if not str(s.meta.get("region") or "").startswith("chrome_")]
    assert negs
    expanded = prose.bbox.expand(Q1_NEG_CLEARANCE_PX)
    for s in negs:
        x, y = s.meta["point"]
        assert not expanded.contains(x, y)


def test_emit_marker_area_frac_is_random_in_range(tmp_path: Path):
    """Current protocol: each marked image draws 0.08%-0.2% of the page area."""
    from point_ocr.marker import MARKER_AREA_FRAC_MAX, MARKER_AREA_FRAC_MIN

    img = Image.new("RGB", (900, 700), (245, 245, 240))
    blk = BlockAnno(
        "p1",
        BBox(80, 80, 820, 500),
        markdown="A sufficiently long paragraph so the block qualifies as a target.",
        rects=[BBox(80, 80, 820, 500)],
    )
    samples = emit_pool_samples(
        img,
        [blk],
        pool_id="core_inner",
        page_id="t_frac",
        marked_dir=tmp_path / "marked_frac",
        seed=3,
        template_stem="01_article_twocol",
        prompt_key="a2_v3",
    )
    fracs = [float(s.meta["marker_area_frac"]) for s in samples]
    assert fracs
    assert all(MARKER_AREA_FRAC_MIN <= f <= MARKER_AREA_FRAC_MAX for f in fracs)
    assert len(set(fracs)) > 1
    assert all(s.meta.get("marker") == "x45r" for s in samples)

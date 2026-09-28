"""Tests for A2 desktop visibility / focused-window selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from point_ocr.a2_labels import (
    find_enclosing_table_html,
    is_block_visibly_clear,
    select_desktop_positive_blocks,
)
from point_ocr.sample_points import BBox


@dataclass
class _B:
    block_id: str
    bbox: BBox
    markdown: str = "x" * 40
    extra: dict = field(default_factory=dict)
    rects: list = field(default_factory=list)

    def ink_rects(self):
        return list(self.rects) if self.rects else [self.bbox]

    def ink_area(self) -> float:
        return sum(r.area() for r in self.ink_rects())


def test_block_fully_in_frame_rejects_cropped():
    from point_ocr.a2_labels import block_fully_in_frame

    ok = _B("a", BBox(10, 10, 100, 80), extra={"visible_frac": 1.0, "in_viewport_frac": 1.0})
    assert block_fully_in_frame(ok, 200, 200)

    cropped = _B("b", BBox(10, 150, 100, 250), extra={"visible_frac": 1.0})
    assert not block_fully_in_frame(cropped, 200, 200)

    low_vf = _B("c", BBox(10, 10, 100, 80), extra={"visible_frac": 0.5, "in_viewport_frac": 1.0})
    assert not block_fully_in_frame(low_vf, 200, 200)

    low_ivf = _B("d", BBox(10, 10, 100, 80), extra={"visible_frac": 1.0, "in_viewport_frac": 0.7})
    assert not block_fully_in_frame(low_ivf, 200, 200)


def test_block_fully_in_frame_multi_frag_ignores_union_visible_frac():
    from point_ocr.a2_labels import block_fully_in_frame

    wrapped = _B(
        "p",
        BBox(10, 10, 180, 80),
        extra={"visible_frac": 0.32, "in_viewport_frac": 1.0},
        rects=[BBox(10, 10, 90, 40), BBox(100, 10, 180, 40)],
    )
    assert block_fully_in_frame(wrapped, 200, 200)

    cropped_col = _B(
        "q",
        BBox(10, 10, 180, 250),
        extra={"visible_frac": 0.32, "in_viewport_frac": 0.6},
        rects=[BBox(10, 10, 90, 40), BBox(100, 150, 180, 250)],
    )
    assert not block_fully_in_frame(cropped_col, 200, 200)


def test_table_cell_promotes_to_full_table():
    html = """
    <html><body>
    <table><tr><th data-block-id="h0">A</th></tr>
    <tr><td data-block-id="c0">1</td></tr></table>
    </body></html>
    """
    got = find_enclosing_table_html(html, "c0")
    assert got is not None
    assert got.startswith("<table>")
    assert "data-block-id" not in got
    assert ">1<" in got and ">A<" in got


def test_semantic_group_rewrite_expands_short_seed():
    from point_ocr.build_point import BlockAnno
    from point_ocr.a2_labels import rewrite_block_for_a2, build_semantic_group_map

    html = """
    <html><body>
    <aside data-semantic-group="hot" data-group-kind="heading_list">
      <p data-block-id="h" data-role="seed">热榜</p>
      <p data-block-id="i1" data-role="member">1. item</p>
      <p data-block-id="i2" data-role="member">2. item</p>
    </aside>
    </body></html>
    """
    seed = BlockAnno("h", BBox(0, 0, 40, 20), markdown="热榜", extra={"tag": "p"})
    gmap = build_semantic_group_map(html)
    out = rewrite_block_for_a2(seed, html, gmap)
    assert out.extra.get("a2_kind") == "semantic_group"
    assert "热榜" in out.markdown and "1. item" in out.markdown
    assert out.extra.get("group_role") == "seed"

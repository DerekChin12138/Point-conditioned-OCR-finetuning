"""Tests for static POINT prompt protocol."""

from __future__ import annotations

from point_ocr.dataset_format import PointSample
from point_ocr.prompts import (
    PAGE_PROMPT,
    POINT_PROMPT,
    POINT_PROMPT_A1_V2,
    POINT_PROMPT_A2_V2,
    POINT_PROMPT_A2_V3,
    POINT_PROMPT_OCR_MT_V1,
    get_prompt,
    pixel_to_norm,
)


def test_point_prompt_a1_is_minimal_block():
    low = POINT_PROMPT_A1_V2.lower()
    assert "magenta" in low
    assert "45" in POINT_PROMPT_A1_V2
    assert "x-shaped" in low
    assert "minimal semantic block" in low
    assert "nearby group" not in low
    assert "empty string" in low
    assert "x=" not in POINT_PROMPT_A1_V2


def test_point_prompt_a2_allows_group():
    low = POINT_PROMPT_A2_V2.lower()
    assert "magenta" in low
    assert "45" in POINT_PROMPT_A2_V2
    assert "x-shaped" in low
    assert "nearby group" in low
    assert "table" in low


def test_point_prompt_a2_v3_is_short():
    low = POINT_PROMPT_A2_V3.lower()
    assert "magenta-and-white cross" in low
    assert "45" in POINT_PROMPT_A2_V3
    assert "most related" in low
    assert "nearby group" in low
    assert "format tables" not in low
    assert "do not output the full page" in low
    assert "transcribe formulas and code" in low
    assert "a table" in low or "table" in low
    assert "formula, code, or image" not in low


def test_default_point_prompt_is_a2_v3():
    assert POINT_PROMPT == POINT_PROMPT_A2_V3
    assert get_prompt("POINT") == POINT_PROMPT_A2_V3


def test_get_prompt_prompt_key():
    assert get_prompt("POINT", prompt_key="a1_v2") == POINT_PROMPT_A1_V2
    assert get_prompt("POINT", prompt_key="a2_v2") == POINT_PROMPT_A2_V2
    assert get_prompt("POINT", prompt_key="a2_v3") == POINT_PROMPT_A2_V3
    assert get_prompt("POINT", prompt_key="ocr_mt_v1") == POINT_PROMPT_OCR_MT_V1
    assert get_prompt("PAGE") == PAGE_PROMPT
    from point_ocr.prompts import (
        POINT_PROMPT_COORD_Q1_TEMPLATE,
        POINT_PROMPT_COORD_V1_TEMPLATE,
        format_coord_prompt_q1,
        format_coord_prompt_v1,
        format_point_2d_json,
        format_point_tag,
        pixel_to_pct_topleft,
    )

    assert get_prompt("POINT", prompt_key="coord_v1") == POINT_PROMPT_COORD_V1_TEMPLATE
    assert get_prompt("POINT", prompt_key="coord_q1") == POINT_PROMPT_COORD_Q1_TEMPLATE
    assert get_prompt("POINT", prompt_key="q1_coordinate") == POINT_PROMPT_COORD_Q1_TEMPLATE
    filled = format_coord_prompt_v1(960, 540, 1920, 1080)
    assert "x=500" in filled and "y=500" in filled
    q1 = format_coord_prompt_q1(960, 540, 1920, 1080)
    assert "no mark" not in q1.lower()
    assert '"point_2d": [960, 540]' in q1
    assert format_point_2d_json(32.4, 31.6) == '[{"point_2d": [32, 32]}]'
    assert pixel_to_pct_topleft(0, 0, 100, 200) == (0.0, 0.0)
    assert format_point_tag(32.4, 31.6) == "<point>\nx: 32.4%\ny: 31.6%\n</point>"


def test_pixel_to_norm_corners():
    assert pixel_to_norm(0, 0, 1920, 1080) == (0, 0)
    nx, ny = pixel_to_norm(1919, 1079, 1920, 1080)
    assert 990 <= nx < 1000
    assert 990 <= ny < 1000


def test_point_sample_user_text_respects_prompt_key():
    s = PointSample(
        sample_id="t",
        image_path="/tmp/x.jpg",
        task="POINT",
        target="hello",
        meta={"prompt_key": "a2_v2"},
    )
    assert s.user_text() == POINT_PROMPT_A2_V2
    s2 = PointSample(
        sample_id="t2",
        image_path="/tmp/x.jpg",
        task="POINT",
        target="hello",
        meta={"prompt_key": "a1_v2"},
    )
    assert s2.user_text() == POINT_PROMPT_A1_V2
    s3 = PointSample(
        sample_id="t3",
        image_path="/tmp/x.jpg",
        task="POINT",
        target="hello",
        meta={"prompt_key": "a2_v3"},
    )
    assert s3.user_text() == POINT_PROMPT_A2_V3

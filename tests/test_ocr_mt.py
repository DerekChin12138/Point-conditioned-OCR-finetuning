"""OCR-MT XML target, prompt key, and unique-within-page bilingual fill."""

from __future__ import annotations

import random
import sys
from pathlib import Path

from point_ocr.ocr_mt import (
    format_ocr_mt_target,
    is_ocr_mt_prompt,
    parse_ocr_mt_target,
    wrap_point_target,
)
from point_ocr.pools.spec import quota_from_frac
from point_ocr.prompts import POINT_PROMPT_OCR_MT_V1, get_prompt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from apply_content_pack import UniquePicker, fill_from_pool  # noqa: E402
from point_ocr.pools.compose import load_recipe


def test_ocr_mt_prompt_registered():
    assert "source" in POINT_PROMPT_OCR_MT_V1.lower()
    assert "translation" in POINT_PROMPT_OCR_MT_V1.lower()
    assert get_prompt("POINT", prompt_key="ocr_mt_v1") == POINT_PROMPT_OCR_MT_V1
    assert is_ocr_mt_prompt("ocr_mt_v1")
    assert not is_ocr_mt_prompt("a2_v3")


def test_xml_roundtrip_and_empty():
    xml = format_ocr_mt_target("Hello there.", "你好。")
    assert parse_ocr_mt_target(xml) == ("Hello there.", "你好。")
    wrapped = wrap_point_target("Hello there.", prompt_key="ocr_mt_v1", translation="你好。")
    assert wrapped == xml
    assert wrap_point_target("Hello", prompt_key="a2_v3", translation="你好") == "Hello"
    assert wrap_point_target("Hello", prompt_key="ocr_mt_v1", translation="x", is_negative=True) == ""
    assert wrap_point_target("", prompt_key="ocr_mt_v1", translation="x") == ""


def test_unique_picker_no_repeat_on_page():
    pool = {
        "paragraphs": [
            {"en": "Alpha sentence one.", "zh": "甲。", "domain": "daily_dialogue"},
            {"en": "Bravo sentence two.", "zh": "乙。", "domain": "daily_dialogue"},
            {"en": "Charlie sentence three.", "zh": "丙。", "domain": "spec_legal"},
        ],
        "headings": [{"en": "Title A", "zh": "标题甲", "domain": "daily_mixed"}],
    }
    picker = UniquePicker(pool, random.Random(0))
    seen = [picker.pick("paragraphs") for _ in range(3)]
    assert len(set(seen)) == 3


def test_fill_sets_data_mt_zh():
    html = """<html><body>
    <p data-block-id="p1">placeholder long enough for paragraph bucket xxxxxxxxxxxx</p>
    <p data-block-id="p2">another placeholder long enough for paragraph bucket yyyyyyyyyyyy</p>
    </body></html>"""
    pool = {
        "paragraphs": [
            {"en": "The train is late again this morning.", "zh": "今天早上火车又晚点了。", "domain": "daily_dialogue"},
            {"en": "Please close the window before you leave.", "zh": "离开前请关上窗户。", "domain": "daily_dialogue"},
            {"en": "Dinner will be ready in twenty minutes.", "zh": "晚饭二十分钟后就好。", "domain": "daily_dialogue"},
            {"en": "We should buy tickets in advance.", "zh": "我们应该提前买票。", "domain": "daily_dialogue"},
        ]
    }
    filled, _ = fill_from_pool(html, pool, random.Random(1), extra_paragraphs=0, occupancy_band="occ_40")
    assert "data-mt-zh" in filled
    assert "placeholder long enough" not in filled
    # page-unique English
    assert filled.count("The train is late again this morning.") <= 1


def test_q2_recipe_is_double_q1_mix():
    q1 = load_recipe(ROOT / "data/recipes/stage_q1_withreal.yaml")
    q2 = load_recipe(ROOT / "data/recipes/stage_q2_ocr_mt.yaml")
    assert q2["n"] == 2 * q1["n"]
    assert q2["prompt_key"] == "ocr_mt_v1"
    assert q1["mix"] == q2["mix"]
    q = quota_from_frac(q2["mix"], q2["n"])
    assert q == {
        "core_inner": 30000,
        "empty_clear": 7500,
        "empty_special": 5000,
        "multi_frag": 7500,
        "semantic_group": 3750,
    }

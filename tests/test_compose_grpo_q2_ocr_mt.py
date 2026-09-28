"""GRPO_Q2 filtering: two-axis "good" (source hit + chrF) value scoring."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from compose_grpo_q2_ocr_mt import is_selectable, score_record  # noqa: E402

GT = "<source>\nHello world.\n</source>\n<translation>\n你好世界。\n</translation>"
DUMP = "<source>\n" + ("lorem ipsum dolor " * 60) + "\n</source>\n<translation>\n很长的垃圾翻译。\n</translation>"
HALLUC = "<source>\nInvented block.\n</source>\n<translation>\n编造的块。\n</translation>"
CRITERIA = dict(min_good_frac=0.125, max_good_frac=0.875, min_std=0.03)
KW = dict(source_threshold=0.85, translation_threshold=0.5, bad_translation_chrf=0.3)


def _rec(completions, target=GT, negative=False):
    return {"completions": completions, "target": target, "is_negative": negative}


def test_positive_mixed_group_selectable():
    rec = score_record(_rec([GT, GT, DUMP, ""]), **KW)
    assert rec["frac_good"] == 0.5
    assert rec["frac_over"] >= 0.25 and rec["frac_bad_translation"] >= 0.25
    assert rec["value"] > 0
    assert is_selectable(rec, **CRITERIA)


def test_saturated_and_collapsed_positive_dropped():
    sat = score_record(_rec([GT, GT, GT, GT]), **KW)
    assert sat["frac_good"] == 1.0 and not is_selectable(sat, **CRITERIA)
    col = score_record(_rec(["", "", "", ""]), **KW)
    assert col["frac_good"] == 0.0 and not is_selectable(col, **CRITERIA)


def test_negative_split_selectable():
    rec = score_record(_rec(["", "", HALLUC, HALLUC], target="", negative=True), **KW)
    assert rec["frac_good"] == 0.5
    assert rec["value"] == 1.0
    assert is_selectable(rec, **CRITERIA)


def test_negative_all_empty_dropped():
    rec = score_record(_rec(["", "", "", ""], target="", negative=True), **KW)
    assert rec["frac_good"] == 1.0 and not is_selectable(rec, **CRITERIA)

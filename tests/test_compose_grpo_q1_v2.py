"""GRPO_Q1 v2 filtering: reward-scale-aware value scoring."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from compose_grpo_q1_v2 import is_selectable, score_record  # noqa: E402

GT = "The mediator wrote it on a napkin. Both signed. The office filed it as an official agreement."
DUMP = "# Block A\n\n" + ("lorem ipsum " * 80)
PARTIAL = "The mediator wrote it on a napkin."
CRITERIA = dict(min_good_frac=0.125, max_good_frac=0.875, min_std=0.03)


def _rec(completions, target=GT, negative=False):
    return {"completions": completions, "target": target, "is_negative": negative}


def test_positive_mixed_group_is_selectable_and_boundary_valued():
    rec = score_record(_rec([GT, GT, DUMP, PARTIAL]), good_frac=0.8)
    assert rec["ceiling"] == 1.0
    assert abs(rec["frac_good"] - 0.5) < 1e-9
    assert rec["frac_over"] == 0.25 and rec["frac_under"] == 0.25
    assert abs(rec["value"] - 0.5) < 1e-9
    assert is_selectable(rec, **CRITERIA)


def test_saturated_positive_is_dropped():
    rec = score_record(_rec([GT, GT, GT, GT]), good_frac=0.8)
    assert rec["frac_good"] == 1.0
    assert rec["reward_std"] == 0.0
    assert not is_selectable(rec, **CRITERIA)


def test_collapsed_positive_is_dropped():
    rec = score_record(_rec(["", "", "", ""]), good_frac=0.8)
    assert rec["frac_good"] == 0.0
    assert not is_selectable(rec, **CRITERIA)


def test_negative_split_group_is_selectable():
    halluc = "Some invented block text that should not be here."
    rec = score_record(_rec(["", "", halluc, halluc], target="", negative=True), good_frac=0.8)
    assert rec["ceiling"] == 2.5  # edit + empty weight
    assert abs(rec["frac_good"] - 0.5) < 1e-9
    assert abs(rec["value"] - 1.0) < 1e-9
    assert is_selectable(rec, **CRITERIA)


def test_negative_all_empty_is_dropped():
    rec = score_record(_rec(["", "", "", ""], target="", negative=True), good_frac=0.8)
    assert rec["frac_good"] == 1.0
    assert not is_selectable(rec, **CRITERIA)


def test_overs_dumps_rank_above_partial_only():
    mostly_over = score_record(_rec([GT, DUMP, DUMP, DUMP]), good_frac=0.8)
    mostly_under = score_record(_rec([GT, PARTIAL, PARTIAL, PARTIAL]), good_frac=0.8)
    # both carry boundary signal; over-dump weight equals under-miss weight
    assert mostly_over["value"] > 0 and mostly_under["value"] > 0
    assert is_selectable(mostly_over, **CRITERIA) and is_selectable(mostly_under, **CRITERIA)

"""Q2 GRPO reward prototype: per-term values and ranking sanity."""

from __future__ import annotations

from point_ocr.grpo_rewards_q2 import (
    DEFAULT_WEIGHTS,
    empty_reward,
    hallucination_penalty,
    over_extraction_penalty,
    reward_breakdown,
    source_reward,
    translation_reward,
    weighted_rewards_q2,
    xml_reward,
)

GT = "<source>\nHello world.\n</source>\n<translation>\n你好世界。\n</translation>"
GT_LONG = "<source>\n" + ("A long paragraph sentence. " * 20) + "\n</source>\n<translation>\n很长的段落。\n</translation>"


def test_positive_terms():
    assert source_reward([GT], [GT], [False]) == [1.0]
    assert translation_reward([GT], [GT], [False]) == [1.0]
    assert xml_reward([GT]) == [1.0]
    assert empty_reward([GT], [GT], [False]) == [0.0]
    assert hallucination_penalty([GT], [GT], [False]) == [0.0]


def test_positive_empty_is_penalised():
    assert empty_reward([""], [GT], [False]) == [-1.0]
    assert source_reward([""], [GT], [False]) == [0.0]
    assert translation_reward([""], [GT], [False]) == [0.0]


def test_negative_empty_variants():
    assert empty_reward([""], [""], [True]) == [1.0]
    assert empty_reward(["<source>\n</source>\n<translation>\n</translation>"], [""], [True]) == [0.5]
    assert empty_reward(["<source>"], [""], [True]) == [0.5]
    assert hallucination_penalty(["<source>"], [""], [True]) == [0.0]


def test_negative_hallucination_penalised():
    halluc = "<source>\nLeaked text.\n</source>\n<translation>\n泄漏文本。\n</translation>"
    assert empty_reward([halluc], [""], [True]) == [0.0]
    assert hallucination_penalty([halluc], [""], [True]) == [-1.0]
    assert xml_reward([halluc]) == [1.0]  # structurally fine, semantically wrong


def test_plain_text_structure_penalty():
    assert xml_reward(["just some markdown text"]) == [-0.5]
    assert xml_reward([""]) == [0.0]


def test_over_extraction_penalty():
    dump = "<source>\n" + ("Another block. " * 60) + "\n</source>\n<translation>\n另一个块。\n</translation>"
    assert over_extraction_penalty([dump], [GT], [False]) == [-1.0]
    assert over_extraction_penalty([GT], [GT], [False]) == [0.0]


def test_weighted_ranking_positives():
    wrong = "<source>\nCompletely different.\n</source>\n<translation>\n完全不同。\n</translation>"
    empty = ""
    rewards = weighted_rewards_q2([GT, wrong, empty], [GT, GT, GT], [False, False, False])
    assert rewards[0] > rewards[1] > rewards[2]


def test_weighted_ranking_negatives():
    halluc = "<source>\nLeaked.\n</source>\n<translation>\n泄漏。\n</translation>"
    rewards = weighted_rewards_q2(["", halluc], ["", ""], [True, True])
    assert rewards[0] > rewards[1]


def test_reward_breakdown_shape():
    rows = reward_breakdown([GT], [GT], [False])
    assert len(rows) == 1
    assert set(rows[0]) >= {"source", "translation", "xml", "empty", "hallucination", "over_extraction", "leak", "total"}
    assert DEFAULT_WEIGHTS.empty > DEFAULT_WEIGHTS.source


def test_target_list_and_bool_negative():
    # A scalar is_negative flag is broadcast: both rows are negatives, so the
    # one that stayed empty must beat the one that emitted a source.
    rewards = weighted_rewards_q2([GT, ""], [GT, ""], True)
    assert len(rewards) == 2
    assert rewards[1] > rewards[0]

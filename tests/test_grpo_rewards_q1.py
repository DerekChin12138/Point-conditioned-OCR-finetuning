"""Q1 GRPO reward v2: boundary terms for multi-block / short-group / empty."""

from __future__ import annotations

from point_ocr.grpo_rewards_q1 import (
    DEFAULT_WEIGHTS,
    edit_reward,
    empty_reward,
    over_extraction_penalty,
    reward_breakdown,
    under_extraction_penalty,
    weighted_rewards,
)

GT = "The mediator wrote it on a napkin. Both signed. The office filed it as an official agreement."
GROUP = "## Renfield will go straight to Dracula. Operations against FDLR had been accompanied by an increase in threats."
DUMP = ("# Block A\n\n" + ("lorem ipsum " * 80) + "\n\n# Block B\n\n" + ("dolor sit " * 80))


def test_edit_and_empty_terms():
    assert edit_reward([GT], [GT], [False]) == [1.0]
    assert empty_reward([GT], [GT], [False]) == [0.0]
    assert empty_reward([""], [GT], [False]) == [-1.0]
    assert empty_reward([""], [""], [True]) == [1.0]
    assert empty_reward([GT], [""], [True]) == [0.0]


def test_over_extraction_penalty():
    assert over_extraction_penalty([DUMP], [GT], [False]) == [-1.0]
    assert over_extraction_penalty([GT], [GT], [False]) == [0.0]
    assert over_extraction_penalty([DUMP], [""], [True]) == [0.0]


def test_under_extraction_penalty():
    # strict sub-part of a multi-fragment / group unit
    assert under_extraction_penalty(["## Renfield will go straight to Dracula."], [GROUP], [False]) == [-1.0]
    # complete answer -> no penalty
    assert under_extraction_penalty([GROUP], [GROUP], [False]) == [0.0]
    # empty is handled by empty_reward, not here
    assert under_extraction_penalty([""], [GROUP], [False]) == [0.0]


def test_weighted_ranking_positive():
    partial = "## Renfield will go straight to Dracula."
    rewards = weighted_rewards([GROUP, partial, ""], [GROUP, GROUP, GROUP], [False, False, False])
    assert rewards[0] > rewards[1] > rewards[2]


def test_weighted_ranking_negative():
    halluc = "Some invented block text."
    rewards = weighted_rewards(["", halluc], ["", ""], [True, True])
    assert rewards[0] > rewards[1]


def test_breakdown_and_weights():
    rows = reward_breakdown([GT], [GT], [False])
    assert set(rows[0]) >= {
        "edit",
        "empty",
        "over_extraction",
        "under_extraction",
        "leak",
        "total",
    }
    # empty discipline is weighted above raw localization
    assert DEFAULT_WEIGHTS.empty > DEFAULT_WEIGHTS.edit


def test_reward_func_count_matches_weights():
    from point_ocr.grpo_rewards_q1 import REWARD_FUNCS, REWARD_WEIGHTS

    assert len(REWARD_FUNCS) == len(REWARD_WEIGHTS) == 5

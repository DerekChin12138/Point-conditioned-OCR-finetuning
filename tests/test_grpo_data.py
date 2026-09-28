"""GRPO hard-scene sampling and rewards."""

from __future__ import annotations

import random

from point_ocr.grpo_data import (
    grpo_scene,
    pick_unique_page_first,
    select_grpo_rows,
    select_sliced_rows,
    val_slice,
)
from point_ocr.grpo_rewards import (
    cleaned_pred,
    completion_text,
    edit_reward,
    empty_reward,
    leak_penalty,
    weighted_rewards,
)


def _row(pool: str, page: str, target: str, *, n_frag: int = 1, bboxes=None, chrome="none") -> dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>prompt"},
            {"role": "assistant", "content": target},
        ],
        "images": ["x.jpg"],
        "metadata": {
            "pool_id": pool,
            "page_id": page,
            "sample_id": f"{page}:{target[:8]}",
            "n_fragments": n_frag,
            "bboxes": bboxes or [[0, 0, 10, 10]],
            "chrome_family": chrome,
            "is_negative": target == "",
        },
    }


def test_grpo_scene_mapping():
    assert grpo_scene(_row("multi_frag", "p1", "hello world " * 20)) == "multi_frag"
    assert grpo_scene(_row("core_inner", "p1", "hello world " * 20, n_frag=2)) == "multi_frag"
    assert grpo_scene(_row("semantic_group", "p1", "Short card")) == "semantic_group"
    assert grpo_scene(_row("empty_clear", "p1", "")) == "empty"
    assert grpo_scene(_row("empty_special", "p1", "")) == "empty"
    assert grpo_scene(_row("core_inner", "p1", "A long core paragraph " * 20)) is None
    real_short = _row("real_labeled", "r1", "OK")
    assert grpo_scene(real_short) == "semantic_group"
    real_long = _row("real_labeled", "r2", "A" * 120)
    assert grpo_scene(real_long) is None
    assert val_slice(real_long) == "regular"
    assert val_slice(_row("core_inner", "p1", "A long core paragraph " * 20)) == "regular"
    assert val_slice(_row("empty_clear", "p1", "")) == "empty"


def test_select_val_slices_even_mix():
    rows = []
    rows += [_row("core_inner", f"c{i}", "para " * 30) for i in range(40)]
    rows += [_row("multi_frag", f"m{i}", "para " * 30) for i in range(40)]
    rows += [_row("semantic_group", f"s{i}", "card") for i in range(40)]
    rows += [_row("empty_clear", f"e{i}", "", chrome="browser") for i in range(40)]
    mix = {"regular": 0.25, "multi_frag": 0.25, "semantic_group": 0.25, "empty": 0.25}
    picked = select_sliced_rows(
        rows, 20, mix, random.Random(2), slice_fn=val_slice, stage="grpo_q1_signal"
    )
    assert len(picked) == 20
    scenes = [r["metadata"]["grpo_scene"] for r in picked]
    assert scenes.count("regular") == 5
    assert scenes.count("multi_frag") == 5
    assert scenes.count("semantic_group") == 5
    assert scenes.count("empty") == 5


def test_unique_page_first_covers_pages_before_repeats():
    rows = [_row("multi_frag", f"p{i}", "x", chrome="none") for i in range(5)]
    rows += [_row("multi_frag", "p0", "y", chrome="none") for _ in range(5)]
    picked = pick_unique_page_first(rows, 5, random.Random(0))
    pages = [r["metadata"]["page_id"] for r in picked]
    assert len(set(pages)) == 5


def test_select_grpo_rows_respects_mix():
    rows = []
    rows += [_row("multi_frag", f"m{i}", "para " * 30) for i in range(40)]
    rows += [_row("semantic_group", f"s{i}", "card") for i in range(40)]
    rows += [_row("empty_clear", f"e{i}", "", chrome="browser") for i in range(30)]
    rows += [_row("empty_special", f"t{i}", "", chrome="none") for i in range(20)]
    mix = {"multi_frag": 0.4, "semantic_group": 0.3, "empty": 0.3}
    picked = select_grpo_rows(
        rows,
        20,
        mix,
        random.Random(1),
        stage="grpo_q1_hard",
        empty_pools={"empty_clear": 0.78, "empty_special": 0.22},
        empty_chrome_frac=0.5,
    )
    assert len(picked) == 20
    scenes = [r["metadata"]["grpo_scene"] for r in picked]
    assert scenes.count("multi_frag") == 8
    assert scenes.count("semantic_group") == 6
    assert scenes.count("empty") == 6


def test_keep_all_real_then_fill_synth():
    reals = [_row("real_labeled", f"r{i}", "OK") for i in range(5)]
    reals += [_row("real_labeled", f"rm{i}", "frag", n_frag=2) for i in range(2)]
    synth = []
    synth += [_row("multi_frag", f"m{i}", "para " * 30) for i in range(40)]
    synth += [_row("semantic_group", f"s{i}", "card") for i in range(40)]
    synth += [_row("empty_clear", f"e{i}", "", chrome="browser") for i in range(40)]
    picked = select_grpo_rows(
        reals + synth,
        20,
        {"multi_frag": 0.4, "semantic_group": 0.3, "empty": 0.3},
        random.Random(3),
        stage="grpo_q1_hard",
        keep_all_real=True,
    )
    assert len(picked) == 20
    real_pages = {r["metadata"]["page_id"] for r in reals}
    kept = {r["metadata"]["page_id"] for r in picked if r["metadata"]["pool_id"] == "real_labeled"}
    assert kept == real_pages
    assert sum(1 for r in picked if r["metadata"]["pool_id"] == "real_labeled") == 7


def test_completion_text_unwraps_chat():
    assert completion_text("hello") == "hello"
    assert completion_text([{"role": "assistant", "content": "block"}]) == "block"
    assert (
        completion_text([{"role": "assistant", "content": [{"type": "text", "text": "x"}]}])
        == "x"
    )
    assert cleaned_pred("hello<|im_end|>") == "hello"


def test_grpo_rewards_empty_and_hit():
    completions = ["hello", "", ""]
    targets = ["hello", "", "a full paragraph that should be read"]
    negs = [False, True, False]
    edits = edit_reward(completions, targets)
    assert edits[0] == 1.0
    assert edits[1] == 1.0
    assert edits[2] == 0.0
    empties = empty_reward(completions, targets, is_negative=negs)
    assert empties[0] == 0.0
    assert empties[1] == 1.0
    assert empties[2] == -1.0
    leaks = leak_penalty(["ok", "x<think>y"])
    assert leaks == [0.0, -0.5]


def test_empty_pred_worse_than_short_fragment():
    gt = "The full paragraph that should be transcribed completely from the block."
    empty_r = weighted_rewards([""], gt, is_negative=False)[0]
    short_r = weighted_rewards(["The full paragraph"], gt, is_negative=False)[0]
    hit_r = weighted_rewards([gt], gt, is_negative=False)[0]
    assert empty_r == -1.0
    assert short_r > empty_r
    assert hit_r > short_r


def test_weighted_rewards_and_verdict():
    from point_ocr.grpo_rewards import group_learning_verdict, group_reward_stats, weighted_rewards

    vals = weighted_rewards(["hello", "hello"], "hello", is_negative=False)
    assert vals[0] == 1.0
    stats = group_reward_stats([1.0, 1.0, 1.0])
    assert stats["std"] == 0.0
    assert group_learning_verdict(0.95, 0.0, 1) == "saturated"
    assert group_learning_verdict(0.5, 0.12, 3) == "signal"
    assert group_learning_verdict(0.05, 0.0, 1) == "collapsed"

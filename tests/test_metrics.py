"""Metric tests."""

from __future__ import annotations

from point_ocr.metrics import (
    EvalExample,
    edit_similarity,
    evaluate_examples,
    looks_like_over_extraction,
)


def test_edit_similarity_identity():
    assert edit_similarity("Hello World", "hello   world") == 1.0


def test_over_extraction():
    gt = "Short block."
    pred = "# A\n\n" + ("word " * 300) + "\n\n# B\n\nmore\n\n# C\n\nend"
    assert looks_like_over_extraction(pred, gt)


def test_report_rates():
    examples = [
        EvalExample("1", "hello", "hello", False),
        EvalExample("2", "page dump " * 100, "x", False),
        EvalExample("3", "", "", True),
        EvalExample("4", "oops", "", True),
    ]
    r = evaluate_examples(examples, hit_threshold=0.85)
    assert r.n == 4
    assert r.n_positive == 2
    assert r.n_negative == 2
    assert r.block_hit_rate == 0.5
    assert r.empty_on_chrome_rate == 0.5
    assert r.over_extraction_rate > 0

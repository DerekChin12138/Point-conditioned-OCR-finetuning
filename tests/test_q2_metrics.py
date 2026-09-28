"""Q2 metrics: XML parsing, chrF, and empty-discipline accounting."""

from __future__ import annotations

from point_ocr.q2_metrics import (
    Q2Example,
    chrf,
    evaluate_q2,
    parse_ocr_mt_prediction,
    report_to_markdown,
)


def test_parse_well_formed():
    p = parse_ocr_mt_prediction("<source>\nHello world.\n</source>\n<translation>\n你好世界。\n</translation>")
    assert p.well_formed
    assert p.source == "Hello world."
    assert p.translation == "你好世界。"
    assert p.source_before_translation
    assert not p.empty_shell and not p.truncated


def test_parse_empty_shell_and_truncated():
    shell = parse_ocr_mt_prediction("<source>\n</source>\n<translation>\n</translation>")
    assert shell.empty_shell
    assert shell.well_formed  # structurally valid, just semantically empty
    trunc = parse_ocr_mt_prediction("<source>")
    assert trunc.truncated
    assert not trunc.has_source and not trunc.has_translation


def test_parse_stray_text_not_well_formed():
    p = parse_ocr_mt_prediction(
        "Here you go:\n<source>A</source>\n<translation>甲</translation>"
    )
    assert p.has_source and p.has_translation
    assert not p.well_formed
    assert "Here you go" in p.stray


def test_chrf_bounds():
    assert chrf("你好世界", "你好世界") == 1.0
    assert chrf("", "") == 1.0
    assert chrf("abc", "") == 0.0
    assert chrf("", "abc") == 0.0
    assert 0.0 < chrf("你好世界啊", "你好世界") < 1.0
    # chrF++ averages word n-grams in as well.
    assert chrf("the quick brown fox", "the quick brown fox", word_order=2) == 1.0


def test_evaluate_q2_positive_and_negative():
    pos_ok = Q2Example(
        "p1",
        "<source>\nHello world.\n</source>\n<translation>\n你好世界。\n</translation>",
        "<source>\nHello world.\n</source>\n<translation>\n你好世界。\n</translation>",
    )
    pos_wrong = Q2Example(
        "p2",
        "<source>\nTotally different.\n</source>\n<translation>\n完全不同。\n</translation>",
        "<source>\nHello world.\n</source>\n<translation>\n你好世界。\n</translation>",
    )
    neg_empty = Q2Example("n1", "", "")
    neg_shell = Q2Example("n2", "<source>\n</source>\n<translation>\n</translation>", "")
    neg_trunc = Q2Example("n3", "<source>", "")
    neg_halluc = Q2Example(
        "n4",
        "<source>\nLeaked text.\n</source>\n<translation>\n泄漏文本。\n</translation>",
        "",
    )
    rep = evaluate_q2([pos_ok, pos_wrong, neg_empty, neg_shell, neg_trunc, neg_halluc])
    assert rep.n == 6 and rep.n_positive == 2 and rep.n_negative == 4
    assert rep.xml_pair_rate == 1.0
    assert rep.source_hit_rate == 0.5
    assert rep.both_hit_rate == 0.5
    assert rep.strict_empty_rate == 0.25  # only n1
    assert rep.effective_empty_rate == 0.75  # n1, n2, n3
    assert rep.broken_empty_rate == 0.5  # n2, n3
    assert rep.hallucination_rate == 0.25  # n4
    assert rep.format_leak_rate == 0.0


def test_report_to_markdown_smoke():
    rep = evaluate_q2([Q2Example("n", "", "")])
    md = report_to_markdown({"overall": rep.to_dict(), "by_bucket": {"empty_clear": rep.to_dict()}})
    assert "Q2 (OCR + EN→ZH)" in md
    assert "effective empty" in md

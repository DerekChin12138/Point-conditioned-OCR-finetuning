"""Tests for POINT decode cleanup / format-leak detection."""

from __future__ import annotations

from point_ocr.infer import has_format_leak, strip_format_leak, unwrap_text_tokenizer
from point_ocr.metrics import EvalExample, evaluate_examples


class _FakeTok:
    def get_vocab(self):
        return {"a": 0}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTok()


def test_unwrap_text_tokenizer_from_processor():
    proc = _FakeProcessor()
    assert unwrap_text_tokenizer(proc) is proc.tokenizer
    tok = _FakeTok()
    assert unwrap_text_tokenizer(tok) is tok


def test_strip_keeps_clean_block():
    text = "# 端到端文档解析模型如何服务「指哪读哪」交互"
    out = strip_format_leak(text)
    assert out.format_leak is False
    assert out.cleaned == text


def test_strip_cuts_assistant_and_think_runaway():
    raw = (
        "# 端到端文档解析模型如何服务「指哪读哪」交互\n"
        "assistant\n"
        "<think>\n\n</think>\n\n"
        "记者 李华 · 2026-07-15 · 阅读 8 分钟\n\n"
        "传统整页 OCR 会输出完整 Markdown"
    )
    out = strip_format_leak(raw)
    assert out.format_leak is True
    assert out.cleaned == "# 端到端文档解析模型如何服务「指哪读哪」交互"
    assert has_format_leak(raw)


def test_strip_leading_empty_think():
    raw = "<think>\n\n</think>\n\nHello"
    out = strip_format_leak(raw)
    assert out.cleaned == "Hello"


def test_format_leak_rate_in_report():
    examples = [
        EvalExample(
            "1",
            "# title",
            "# title",
            False,
            raw_prediction="# title\nassistant\n<think>\n",
        ),
        EvalExample("2", "ok", "ok", False, raw_prediction="ok"),
    ]
    r = evaluate_examples(examples)
    assert r.format_leak_rate == 0.5
    assert r.block_hit_rate == 1.0

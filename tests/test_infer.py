"""Tests for POINT decode cleanup / format-leak detection."""

from __future__ import annotations

from point_ocr.infer import (
    generate_point_batch,
    has_format_leak,
    strip_format_leak,
    unwrap_text_tokenizer,
)
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


def test_generate_point_batch_plumbing():
    """Batched path: one generate call per chunk, results in input order, correct length."""
    import torch
    from PIL import Image

    class _InnerTok:
        padding_side = "right"
        eos_token_id = 2
        unk_token_id = 0

        def get_vocab(self):
            return {"a": 0}

        def convert_tokens_to_ids(self, _):
            return 1

    class _Proc:
        def __init__(self):
            self.tokenizer = _InnerTok()
            self.calls = 0

        def apply_chat_template(self, messages, **kw):
            return "<image>prompt"

        def __call__(self, images=None, text=None, padding=False, return_tensors=None):
            self.calls += 1
            n = len(text)
            ids = torch.ones((n, 3), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def decode(self, ids, skip_special_tokens=True):
            return "block"

    class _Model:
        def __init__(self):
            self._p = torch.nn.Parameter(torch.zeros(1))
            self.n_generate = 0

        def parameters(self):
            yield self._p

        def generate(self, input_ids=None, **kw):
            self.n_generate += 1
            gen = torch.zeros((input_ids.shape[0], 2), dtype=torch.long)
            return torch.cat([input_ids, gen], dim=1)

    proc, model = _Proc(), _Model()
    images = [Image.new("RGB", (100 + 50 * i, 100)) for i in range(5)]

    out = generate_point_batch(model, proc, images, None, batch_size=8)
    assert len(out) == 5
    assert all(o.cleaned == "block" for o in out)
    assert model.n_generate == 1  # 5 images fit one chunk
    assert proc.tokenizer.padding_side == "left"

    model.n_generate = 0
    out2 = generate_point_batch(model, proc, images, None, batch_size=2)
    assert len(out2) == 5
    assert model.n_generate == 3  # ceil(5/2)

    assert generate_point_batch(model, proc, [], None) == []


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

"""VL on-policy distillation shim: collator/normalisation/eos shim (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import unsloth_gkd_vl as gkd  # noqa: E402


class _FakeTok:
    eos_token = "<|im_end|>"
    eos_token_id = 7
    pad_token_id = 0

    def __init__(self):
        self._vocab = {"<|im_end|>": 7, "<|image_pad|>": 9, "</source>": 11}

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(str(token))


class _FakeProc:
    tokenizer = _FakeTok()
    eos_token = "<|im_end|>"


def test_patch_eos_placeholder_maps_placeholder_only():
    proc = _FakeProc()
    assert proc.tokenizer.convert_tokens_to_ids("<EOS_TOKEN>") is None
    gkd.patch_eos_placeholder(proc)
    assert proc.tokenizer.convert_tokens_to_ids("<EOS_TOKEN>") == 7      # placeholder resolved
    assert proc.tokenizer.convert_tokens_to_ids("<|im_end|>") == 7       # unchanged
    assert proc.tokenizer.convert_tokens_to_ids("nope") is None          # unchanged
    gkd.patch_eos_placeholder(proc)                                       # idempotent
    assert proc.tokenizer.convert_tokens_to_ids("<EOS_TOKEN>") == 7


def test_stage_rows_normalises_sharegpt_string_content(tmp_path: Path):
    from PIL import Image

    p = tmp_path / "a.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(p)
    rows = [{
        "images": [str(p)],
        "messages": [
            {"role": "user", "content": "<image>The image contains a cross. Find the block."},
            {"role": "assistant", "content": "hello block"},
        ],
    }]
    out = gkd.stage_rows(rows, min_pixels=1024, max_pixels=4096)
    content = out[0]["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "text"]
    assert content[1]["text"].startswith("The image contains")
    assert not content[1]["text"].startswith("<image>")
    assert out[0]["messages"][1]["content"][0]["type"] == "text"


def test_perturb_lora_fills_only_lora_b():
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Parameter(torch.zeros(2, 2))
            self.lora_B = nn.Parameter(torch.zeros(2, 2))

    m = M()
    assert int((m.lora_B == 0).sum()) == 4
    n = gkd.perturb_lora(m, scale=0.05, seed=0)
    assert n == 1
    assert not torch.allclose(m.lora_B, torch.zeros(2, 2))
    assert torch.allclose(m.lora_A, torch.zeros(2, 2))  # A untouched

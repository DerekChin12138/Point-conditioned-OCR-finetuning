"""Unsloth's finetune_vision_layers is a no-op for Qwen3.5; we enforce it ourselves."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from unsloth_stage_a import _freeze_vision_lora  # noqa: E402


class _FakeVL(nn.Module):
    """Mimics base_model.model.{visual,language_model}.<..>.lora_A names."""

    def __init__(self) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.visual = nn.Module()
        self.base_model.model.visual.blocks = nn.Module()
        self.base_model.model.visual.blocks.lora_A = nn.Parameter(torch.zeros(2, 2))
        self.base_model.model.visual.blocks.lora_B = nn.Parameter(torch.zeros(2, 2))
        self.base_model.model.language_model = nn.Module()
        self.base_model.model.language_model.lora_A = nn.Parameter(torch.zeros(2, 2))


def test_freeze_vision_lora_only_touches_vision():
    model = _FakeVL()
    assert all(p.requires_grad for p in model.parameters())
    frozen = _freeze_vision_lora(model)
    assert frozen == 2
    names = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert all("visual" in n for n in names)
    assert model.base_model.model.language_model.lora_A.requires_grad


def test_freeze_vision_lora_idempotent():
    model = _FakeVL()
    assert _freeze_vision_lora(model) == 2
    assert _freeze_vision_lora(model) == 0

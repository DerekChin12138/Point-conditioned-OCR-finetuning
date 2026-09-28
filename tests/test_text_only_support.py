"""Text-only sample support + resolution-cap guards.

Two fixes covered here:

1. Samples without an image (e.g. a translation-only task) must survive the
   data conversion, the batched generator and the post-eval generator. Unsloth's
   collator already accepts mixed image/text batches; our own layers used to
   drop them.
2. Training used to silently cap every image at 768px wide via
   ``UnslothVisionDataCollator(resize="min")`` + ``vision_config.image_size``,
   undoing ``--max-pixels``. ``--collator-resize max`` is now the default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.infer import (  # noqa: E402
    generate_point_batch,
    generate_point_text,
    point_user_content,
    resolve_point_prompt,
)
from unsloth_stage_a import (  # noqa: E402
    estimate_vision_tokens,
    sharegpt_to_unsloth,
    warn_vision_budget,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_point_user_content_image_and_text_only():
    img = point_user_content("do it", has_image=True)
    assert [p["type"] for p in img] == ["image", "text"]
    assert img[1]["text"] == "do it"

    txt = point_user_content("translate", has_image=False)
    assert [p["type"] for p in txt] == ["text"]
    assert all(p["type"] != "image" for p in txt)


def test_resolve_point_prompt_without_image_dims():
    assert resolve_point_prompt(prompt="MT: translate", image_w=None, image_h=None) == "MT: translate"
    # falls back to the default POINT prompt when no explicit prompt is given
    default = resolve_point_prompt(prompt=None, image_w=None, image_h=None)
    assert isinstance(default, str) and default.strip()


# --------------------------------------------------------------------------- #
# fake VL processor / model
# --------------------------------------------------------------------------- #
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
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kw):
        has_image = any(p.get("type") == "image" for p in messages[0]["content"])
        return "<image>prompt" if has_image else "prompt"

    def __call__(self, *args, **kwargs):
        text = kwargs.get("text")
        images = kwargs.get("images")
        if text is None and args:
            if isinstance(args[0], str):  # positional text (text-only single call)
                text = [args[0]]
            else:  # positional (image, text) single call
                images = [args[0]]
                text = [args[1]] if len(args) > 1 else None
        self.calls.append({"text": text, "images": images})
        n = len(text) if isinstance(text, list) else 1
        import torch

        ids = torch.ones((n, 3), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        return "block"


class _Model:
    def __init__(self):
        import torch

        self._p = torch.nn.Parameter(torch.zeros(1))
        self.n_generate = 0

    def parameters(self):
        yield self._p

    def generate(self, input_ids=None, **kw):
        import torch

        self.n_generate += 1
        gen = torch.zeros((input_ids.shape[0], 2), dtype=torch.long)
        return torch.cat([input_ids, gen], dim=1)


# --------------------------------------------------------------------------- #
# text-only / mixed generation
# --------------------------------------------------------------------------- #
def test_generate_point_text_without_image_omits_images():
    proc, model = _Proc(), _Model()
    out = generate_point_text(model, proc, None, "MT: translate this")
    assert out.cleaned == "block"
    assert len(proc.calls) == 1
    assert proc.calls[0]["images"] is None  # no vision tokens for a text-only row


def test_generate_point_text_with_image_passes_image():
    from PIL import Image

    proc, model = _Proc(), _Model()
    out = generate_point_text(model, proc, Image.new("RGB", (64, 64)), "point prompt")
    assert out.cleaned == "block"
    assert proc.calls[0]["images"] is not None


def test_generate_point_batch_mixes_image_and_text_only():
    from PIL import Image

    proc, model = _Proc(), _Model()
    images = [Image.new("RGB", (100, 100)), None, Image.new("RGB", (120, 120))]

    out = generate_point_batch(model, proc, images, ["p1", "translate", "p3"], batch_size=8)
    assert len(out) == 3
    assert all(o.cleaned == "block" for o in out)
    # one chunk, and the processor only saw the two real images
    assert len(proc.calls) == 1
    assert len(proc.calls[0]["images"]) == 2
    assert len(proc.calls[0]["text"]) == 3


def test_generate_point_batch_all_text_only():
    proc, model = _Proc(), _Model()
    out = generate_point_batch(model, proc, [None, None], ["a", "b"], batch_size=8)
    assert len(out) == 2
    assert proc.calls[0]["images"] is None


# --------------------------------------------------------------------------- #
# data conversion keeps text-only rows
# --------------------------------------------------------------------------- #
def _row(messages, images=None):
    r = {"messages": messages}
    if images is not None:
        r["images"] = images
    return r


def _msgs(user="do it", target="ok"):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": target}]


def test_sharegpt_to_unsloth_keeps_text_only_rows(tmp_path: Path):
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (32, 32)).save(p)

    rows = [
        _row(_msgs(), images=[str(p)]),          # image row
        _row(_msgs(user="translate me")),          # text-only row (no "images" key)
        _row(_msgs(), images=[]),                  # text-only row (empty list)
        _row(_msgs(), images=[str(tmp_path / "nope.png")]),  # missing file -> dropped
    ]
    out = sharegpt_to_unsloth(rows, lazy_images=True)
    assert len(out) == 3

    types = [[part["type"] for part in ex["messages"][0]["content"]] for ex in out]
    assert types[0] == ["text", "image"]
    assert types[1] == ["text"]
    assert types[2] == ["text"]
    # image path is resolved for the image row
    assert out[0]["messages"][0]["content"][1]["image"] == str(p.resolve())


# --------------------------------------------------------------------------- #
# resolution budget helpers
# --------------------------------------------------------------------------- #
def test_estimate_vision_tokens():
    # Qwen3.5: patch 16, merge 2 -> 1 token per 1024 px
    assert estimate_vision_tokens(0) == 0
    assert estimate_vision_tokens(1024) == 1
    assert estimate_vision_tokens(1280 * 1280) == 1600
    assert estimate_vision_tokens(2880 * 2880) == 8100


def test_warn_vision_budget(capsys):
    warn_vision_budget(1280 * 1280, 8192)  # comfortable -> no warning
    assert capsys.readouterr().out == ""

    warn_vision_budget(2880 * 2880, 8192)  # ~8100 vision tokens -> warning
    assert "vision tokens" in capsys.readouterr().out

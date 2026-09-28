"""POINT inference helpers: stop discipline + format-leak cleanup.

Batch-1 guardrails for Stage A: the model often emits the correct block first,
then continues into chat/thinking artifacts or page dump. Callers should:
  1. generate with eos + stop_strings and a modest max_new_tokens
  2. clean with strip_format_leak before scoring / product use
  3. report format_leak_rate on the raw string
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

# Default budget for one answer. Q1 (block only) needed ~256; Q2 emits
# <source> + <translation>, so a long block can need ~2-3x more tokens.
# 512 keeps long Q2 blocks from being cut off mid-<translation> (which also
# triggered degenerate repetition in the Q2 SFT eval).
DEFAULT_POINT_MAX_NEW_TOKENS = 512

# Strings that should never appear in a finished POINT answer.
POINT_STOP_STRINGS: tuple[str, ...] = (
    "<|im_end|>",
    "<|im_start|>",
    "<think>",
    "</think>",
    "\nassistant",
    "\nAssistant",
)

# Cut raw decode at the first leak marker (order matters: longer / clearer first).
_LEAK_CUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"<think>", re.IGNORECASE),
    re.compile(r"</think>", re.IGNORECASE),
    re.compile(r"(?m)^\s*assistant\s*$", re.IGNORECASE),
    re.compile(r"\nassistant\b", re.IGNORECASE),
)

# Leading empty think shell sometimes survives decode.
_LEADING_EMPTY_THINK = re.compile(
    r"^\s*<think>\s*</think>\s*",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class CleanedPrediction:
    raw: str
    cleaned: str
    format_leak: bool


def has_format_leak(text: str) -> bool:
    """True if raw generation contains chat/thinking control leakage."""
    if not text:
        return False
    t = text
    if "<think>" in t.lower() or "</think>" in t.lower():
        return True
    if "<|im_start|>" in t or "<|im_end|>" in t:
        return True
    if re.search(r"(?m)^\s*assistant\s*$", t, flags=re.IGNORECASE):
        return True
    if re.search(r"\nassistant\b", t, flags=re.IGNORECASE):
        return True
    return False


def strip_format_leak(text: str) -> CleanedPrediction:
    """Keep only content before the first format-leak marker."""
    raw = text if isinstance(text, str) else str(text)
    s = _LEADING_EMPTY_THINK.sub("", raw).strip()
    leak = has_format_leak(s)
    if not leak:
        cleaned = s.strip()
        return CleanedPrediction(raw=raw, cleaned=cleaned, format_leak=False)

    cut_at = len(s)
    for pat in _LEAK_CUT_PATTERNS:
        m = pat.search(s)
        if m and m.start() < cut_at:
            cut_at = m.start()
    cleaned = s[:cut_at].strip()
    # If everything was leak scaffolding, cleaned may be empty.
    return CleanedPrediction(raw=raw, cleaned=cleaned, format_leak=True)


def unwrap_text_tokenizer(processor_or_tokenizer: Any) -> Any:
    """Qwen3VLProcessor has no get_vocab; stop_strings needs the inner tokenizer."""
    inner = getattr(processor_or_tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "get_vocab"):
        return inner
    if hasattr(processor_or_tokenizer, "get_vocab"):
        return processor_or_tokenizer
    return processor_or_tokenizer


def point_eos_token_ids(tokenizer: Any) -> list[int]:
    """Collect EOS / im_end ids for generate(eos_token_id=...)."""
    tok = unwrap_text_tokenizer(tokenizer)
    ids: list[int] = []
    unk = getattr(tok, "unk_token_id", None)
    for name in ("<|im_end|>", "<|endoftext|>"):
        convert = getattr(tok, "convert_tokens_to_ids", None)
        if convert is None:
            continue
        tid = convert(name)
        if tid is None or tid == unk:
            continue
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    eos = getattr(tok, "eos_token_id", None)
    if isinstance(eos, int) and eos >= 0:
        ids.append(eos)
    elif isinstance(eos, Sequence):
        ids.extend(int(x) for x in eos if x is not None)
    # stable unique
    return list(dict.fromkeys(ids))


from point_ocr.prompts import POINT_PROMPT


def apply_point_chat_template(tokenizer: Any, messages: list[dict], **kwargs: Any) -> str:
    """apply_chat_template with thinking disabled when the processor supports it."""
    kw = dict(kwargs)
    kw.setdefault("add_generation_prompt", True)
    kw.setdefault("tokenize", False)
    # Prefer explicit disable; templates that ignore the kw still work.
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kw)


def resolve_point_prompt(
    *,
    image_w: int | None = None,
    image_h: int | None = None,
    point: tuple[float, float] | list[float] | None = None,
    prompt: str | None = None,
) -> str:
    """Prefer explicit prompt; else static POINT_PROMPT (crosshair is the cue).

    ``image_w``/``image_h`` are optional: a text-only sample (no point crosshair)
    passes ``None`` and only the explicit ``prompt`` is used.
    """
    del image_w, image_h, point  # kept for call-site compatibility
    if prompt is not None and prompt.strip():
        return prompt
    return POINT_PROMPT


def point_user_content(text: str, *, has_image: bool) -> list[dict[str, Any]]:
    """User message parts for a point/vision sample, or a text-only sample.

    The vision collator and Qwen3.5 processor both accept a batch that mixes
    image samples with text-only samples; the text-only parts simply carry no
    image placeholder.
    """
    content: list[dict[str, Any]] = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    return content


def generate_point_text(
    model: Any,
    tokenizer: Any,
    image: Any | None,
    prompt: str | None = None,
    *,
    point: tuple[float, float] | list[float] | None = None,
    max_new_tokens: int = DEFAULT_POINT_MAX_NEW_TOKENS,
    clean: bool = True,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
) -> CleanedPrediction | str:
    """Run POINT generation (greedy by default); return cleaned prediction.

    ``image=None`` runs a **text-only** sample (e.g. the translation half of a
    two-stage pipeline): the prompt is used as-is and no vision tokens are added.
    """
    has_image = image is not None
    if has_image:
        w, h = image.size
    else:
        w, h = 0, 0
    text = resolve_point_prompt(image_w=w, image_h=h, point=point, prompt=prompt)
    messages = [
        {
            "role": "user",
            "content": point_user_content(text, has_image=has_image),
        }
    ]
    input_text = apply_point_chat_template(tokenizer, messages)
    if has_image:
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt")
    else:
        # keyword `text=`: a positional first arg would be parsed as `images`
        # by the Qwen3VL processor.
        inputs = tokenizer(text=input_text, add_special_tokens=False, return_tensors="pt")
    device = next(model.parameters()).device
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    sample = bool(do_sample) and float(temperature) > 0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=int(max_new_tokens),
        use_cache=True,
        do_sample=sample,
    )
    if sample:
        gen_kwargs["temperature"] = max(1e-5, float(temperature))
        gen_kwargs["top_p"] = min(1.0, max(0.01, float(top_p)))
        if top_k is not None and int(top_k) > 0:
            gen_kwargs["top_k"] = int(top_k)
    if repetition_penalty is not None and abs(float(repetition_penalty) - 1.0) > 1e-6:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)
    eos_ids = point_eos_token_ids(tokenizer)
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]

    if seed is not None:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # stop_strings requires a PreTrainedTokenizerBase with get_vocab (not VL Processor).
    text_tok = unwrap_text_tokenizer(tokenizer)
    out_ids = None
    if hasattr(text_tok, "get_vocab"):
        try:
            out_ids = model.generate(
                **inputs,
                **gen_kwargs,
                stop_strings=list(POINT_STOP_STRINGS),
                tokenizer=text_tok,
            )
        except (TypeError, AttributeError, ValueError):
            out_ids = None
    if out_ids is None:
        out_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    gen = out_ids[0][prompt_len:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    cleaned = strip_format_leak(raw)
    return cleaned if clean else cleaned.raw


def _point_gen_kwargs(
    *, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float,
    top_k: int | None, repetition_penalty: float,
) -> dict[str, Any]:
    sample = bool(do_sample) and float(temperature) > 0
    kw: dict[str, Any] = dict(max_new_tokens=int(max_new_tokens), use_cache=True, do_sample=sample)
    if sample:
        kw["temperature"] = max(1e-5, float(temperature))
        kw["top_p"] = min(1.0, max(0.01, float(top_p)))
        if top_k is not None and int(top_k) > 0:
            kw["top_k"] = int(top_k)
    if repetition_penalty is not None and abs(float(repetition_penalty) - 1.0) > 1e-6:
        kw["repetition_penalty"] = float(repetition_penalty)
    return kw


def generate_point_batch(
    model: Any,
    tokenizer: Any,
    images: Sequence[Any],
    prompts: Sequence[str | None] | None = None,
    *,
    max_new_tokens: int = DEFAULT_POINT_MAX_NEW_TOKENS,
    batch_size: int = 8,
    sort_by_area: bool = True,
    clean: bool = True,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
) -> list[Any]:
    """Batched POINT generation. Same prompt/decoding as :func:`generate_point_text`.

    Greedy batch-1 generation leaves the GPU idle (latency-bound).  Batching 8
    similar-sized images is ~4x faster on an 8GB laptop GPU and is the default
    for post-train eval.  Images are sorted by area so a batch pads as little as
    possible.  Returns one :class:`CleanedPrediction` per input, in input order.

    Note: batched greedy decode is not bit-identical to batch-1 (float reduction
    order differs); use the same batching for every model you compare.
    """
    n = len(images)
    if n == 0:
        return []
    if prompts is None:
        prompts = [None] * n
    if len(prompts) != n:
        raise ValueError("images and prompts must have the same length")

    # VL processors expand <image> placeholders; decoder-only generation needs
    # left padding so every prompt ends at the same position.
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if hasattr(inner, "padding_side"):
        inner.padding_side = "left"

    texts: list[str] = []
    for img, prompt in zip(images, prompts):
        has_image = img is not None
        w, h = img.size if has_image else (0, 0)
        text = resolve_point_prompt(image_w=w, image_h=h, prompt=prompt)
        messages = [
            {
                "role": "user",
                "content": point_user_content(text, has_image=has_image),
            }
        ]
        texts.append(apply_point_chat_template(tokenizer, messages))

    order = list(range(n))
    if sort_by_area:
        # text-only samples sort first (area 0); they cost no vision tokens.
        order.sort(
            key=lambda i: (images[i].size[0] * images[i].size[1]) if images[i] is not None else 0
        )

    if seed is not None:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    eos_ids = point_eos_token_ids(tokenizer)
    text_tok = unwrap_text_tokenizer(tokenizer)
    can_stop = hasattr(text_tok, "get_vocab")
    device = next(model.parameters()).device
    bs = max(1, int(batch_size))
    results: list[Any] = [None] * n

    for start in range(0, n, bs):
        idx = order[start : start + bs]
        # Drop None (text-only) entries; the processor matches the remaining
        # images to <image> placeholders in text order.
        chunk_imgs = [images[i] for i in idx if images[i] is not None]
        chunk_texts = [texts[i] for i in idx]
        proc_kwargs: dict[str, Any] = dict(text=chunk_texts, padding=True, return_tensors="pt")
        if chunk_imgs:
            proc_kwargs["images"] = chunk_imgs
        inputs = tokenizer(**proc_kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        gen_kwargs = _point_gen_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
        out_ids = None
        if can_stop:
            try:
                out_ids = model.generate(
                    **inputs, **gen_kwargs, stop_strings=list(POINT_STOP_STRINGS), tokenizer=text_tok
                )
            except (TypeError, AttributeError, ValueError):
                out_ids = None
        if out_ids is None:
            out_ids = model.generate(**inputs, **gen_kwargs)
        for row, i in enumerate(idx):
            raw = tokenizer.decode(out_ids[row][prompt_len:], skip_special_tokens=True).strip()
            parsed = strip_format_leak(raw)
            results[i] = parsed if clean else parsed.raw
    return results

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

# Default budget for a single semantic block (not full-page OCR).
DEFAULT_POINT_MAX_NEW_TOKENS = 256

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


def generate_point_text(
    model: Any,
    tokenizer: Any,
    image: Any,
    prompt: str,
    *,
    max_new_tokens: int = DEFAULT_POINT_MAX_NEW_TOKENS,
    clean: bool = True,
) -> CleanedPrediction | str:
    """Run greedy POINT generation; return cleaned prediction by default."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    input_text = apply_point_chat_template(tokenizer, messages)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt")
    device = next(model.parameters()).device
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
    )
    eos_ids = point_eos_token_ids(tokenizer)
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]

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

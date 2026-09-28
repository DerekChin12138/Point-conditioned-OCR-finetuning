"""GRPO rewards for point-conditioned OCR.

Dense scores so a group of G samples can mix hit / fragment / neighbor / empty.
Mismatch vs GT is edit similarity; empty-on-positive is an extra −1; leak is −0.5.
"""

from __future__ import annotations

from typing import Any

from point_ocr.infer import has_format_leak, strip_format_leak
from point_ocr.metrics import edit_similarity, is_empty_pred


def completion_text(completion: Any) -> str:
    """Unwrap TRL conversational or plain-string completions."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if completion.get("type") == "text":
            return str(completion.get("text") or "")
        if "content" in completion:
            return completion_text(completion.get("content"))
        if "text" in completion:
            return str(completion.get("text") or "")
        return ""
    if isinstance(completion, list):
        return "".join(completion_text(item) for item in completion)
    return str(completion)


def cleaned_pred(completion: Any) -> str:
    return strip_format_leak(completion_text(completion)).cleaned


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def edit_reward(completions: list[Any], target: list[Any], **kwargs: Any) -> list[float]:
    """Normalized edit similarity to GT (empty↔empty = 1)."""
    del kwargs
    out: list[float] = []
    for completion, gt in zip(completions, target):
        pred = cleaned_pred(completion)
        out.append(float(edit_similarity(pred, str(gt or ""))))
    return out


# Positive GT + empty pred: worse than a short related fragment (edit ≈ 0.2–0.5).
EMPTY_MISS = -1.0


def empty_reward(
    completions: list[Any],
    target: list[Any],
    is_negative: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Empty-GT: 1 if still empty, 0 if it leaks text. Non-empty GT: -1 if pred empty, else 0."""
    del kwargs
    flags = is_negative if is_negative is not None else [None] * len(completions)
    out: list[float] = []
    for completion, gt, flag in zip(completions, target, flags):
        empty_gt = (flag is not None and _as_bool(flag)) or is_empty_pred(str(gt or ""))
        pred = cleaned_pred(completion)
        if empty_gt:
            out.append(1.0 if is_empty_pred(pred) else 0.0)
        else:
            out.append(EMPTY_MISS if is_empty_pred(pred) else 0.0)
    return out


def leak_penalty(completions: list[Any], **kwargs: Any) -> list[float]:
    """Penalize chat/think control tokens in the raw decode."""
    del kwargs
    out: list[float] = []
    for completion in completions:
        raw = completion_text(completion)
        out.append(-0.5 if has_format_leak(raw) else 0.0)
    return out


REWARD_FUNCS = [edit_reward, empty_reward, leak_penalty]
REWARD_WEIGHTS = [1.0, 1.0, 0.5]


def weighted_rewards(
    completions: list[Any],
    target: list[Any] | str,
    is_negative: list[Any] | bool | None = None,
) -> list[float]:
    """Same linear combination TRL uses: Σ w_i r_i(completion)."""
    n = len(completions)
    if isinstance(target, list):
        targets = [str(t or "") for t in target]
    else:
        targets = [str(target or "")] * n
    if isinstance(is_negative, list):
        flags: list[Any] = list(is_negative)
    else:
        flags = [is_negative] * n
    edits = edit_reward(completions, targets)
    empties = empty_reward(completions, targets, is_negative=flags)
    leaks = leak_penalty(completions)
    w_e, w_z, w_l = REWARD_WEIGHTS
    return [w_e * edits[i] + w_z * empties[i] + w_l * leaks[i] for i in range(n)]


def group_reward_stats(values: list[float]) -> dict[str, float]:
    """Population mean/std over a GRPO group (same as group-normalized advantages)."""
    n = len(values)
    if n == 0:
        return {"n": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    lo = min(values)
    hi = max(values)
    return {
        "n": float(n),
        "mean": float(mean),
        "std": float(var**0.5),
        "min": float(lo),
        "max": float(hi),
        "range": float(hi - lo),
    }


def group_learning_verdict(
    mean: float,
    std: float,
    n_unique: int,
    *,
    min_std: float = 0.04,
    sat: float = 0.92,
    floor: float = 0.20,
) -> str:
    """signal = enough within-group spread for GRPO; otherwise why it is dead."""
    if std >= min_std and n_unique >= 2:
        return "signal"
    if mean >= sat:
        return "saturated"
    if mean <= floor:
        return "collapsed"
    return "flat"

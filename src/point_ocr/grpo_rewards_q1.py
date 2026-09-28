"""GRPO rewards for Q1 POINT — v2, tuned from the Q1 SFT eval.

What the SFT eval said (``checkpoints/q1_withreal/metrics/final_report.json``):
block_hit 87.6%, over-extraction 0.56%, empty-on-chrome 98.7% on the *regular*
val.  The remaining loss is concentrated in the hard scenes:

* ``multi_frag``   hit 75-77%  -> whole paragraph across fragments
* ``semantic_group`` hit 76%   -> tight short group (bullet/anchor + body)
* empty discipline is good on Q1 negatives but fragile on new/chrome pages,
  and the marker is now drawn at a random size per image, so the model has to
  keep the point localisation while the mark changes size.

So v2 keeps the localization term but adds explicit **boundary** terms:

=======================  ======  =============================================
component                weight  shapes
=======================  ======  =============================================
``edit_reward``           1.0    block-level edit similarity (localization)
``empty_reward``          1.5    empty GT -> stay empty; positive GT -> never empty
``over_extraction_penalty``1.0   dumping several blocks / the whole page
``under_extraction_penalty``1.0  emitting only part of a multi-frag / group unit
``leak_penalty``          0.5    chat/think control-token leakage
=======================  ======  =============================================

Design notes
------------
* **Empty gets the largest weight.**  It is the axis that generalises worst to
  new pages and chrome, and the one a wrong reward most easily destroys.
* **Over/under extraction are separate penalties**, not folded into the hit
  flag: a group of G=8 rollouts then still ranks "missed one member" against
  "dumped four blocks", which is exactly the multi_frag / semantic_group edge.
* **Marker size is not rewarded.**  Size variation now lives in the SFT data
  (uniform 0.08%-0.2% per image), so GRPO must not spend capacity on it.
* Penalties are dense-ish ([-1, 0]) and gated to positives, so negatives are
  decided purely by ``empty_reward``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from point_ocr.grpo_rewards import cleaned_pred, completion_text
from point_ocr.infer import has_format_leak
from point_ocr.metrics import edit_similarity, is_empty_pred, looks_like_over_extraction

UNDER_EXTRACTION_SIM = 0.55
UNDER_EXTRACTION_LEN = 0.65


@dataclass(frozen=True)
class Q1RewardWeights:
    edit: float = 1.0
    empty: float = 1.5
    over_extraction: float = 1.0
    under_extraction: float = 1.0
    leak: float = 0.5


DEFAULT_WEIGHTS = Q1RewardWeights()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _paired(completions: list[Any], target: Any, is_negative: Any):
    n = len(completions)
    if isinstance(target, list):
        targets = [str(t or "") for t in target]
    else:
        targets = [str(target or "")] * n
    if isinstance(is_negative, list):
        flags = [_as_bool(v) for v in is_negative]
        if len(flags) < n:
            flags.extend([False] * (n - len(flags)))
        flags = flags[:n]
    elif is_negative is not None:
        flags = [_as_bool(is_negative)] * n
    else:
        flags = [not t.strip() for t in targets]
    for completion, gt, neg in zip(completions, targets, flags):
        yield cleaned_pred(completion), gt, neg


def edit_reward(completions: list[Any], target: Any, is_negative: Any = None, **kwargs: Any) -> list[float]:
    del kwargs
    return [float(edit_similarity(pred, gt)) for pred, gt, _ in _paired(completions, target, is_negative)]


def empty_reward(completions: list[Any], target: Any, is_negative: Any = None, **kwargs: Any) -> list[float]:
    """Empty GT: +1 if still empty. Positive GT: -1 if the model emitted nothing."""
    del kwargs
    out: list[float] = []
    for pred, _gt, neg in _paired(completions, target, is_negative):
        if neg:
            out.append(1.0 if is_empty_pred(pred) else 0.0)
        else:
            out.append(-1.0 if is_empty_pred(pred) else 0.0)
    return out


def over_extraction_penalty(
    completions: list[Any], target: Any, is_negative: Any = None, **kwargs: Any
) -> list[float]:
    """Positive GT: prediction dumps neighbours / several blocks / the page -> -1."""
    del kwargs
    out: list[float] = []
    for pred, gt, neg in _paired(completions, target, is_negative):
        out.append(-1.0 if (not neg and looks_like_over_extraction(pred, gt)) else 0.0)
    return out


def under_extraction_penalty(
    completions: list[Any], target: Any, is_negative: Any = None, **kwargs: Any
) -> list[float]:
    """Positive GT: prediction is a strict sub-part of the unit (missed members) -> -1."""
    del kwargs
    out: list[float] = []
    for pred, gt, neg in _paired(completions, target, is_negative):
        if neg or is_empty_pred(pred) or not gt.strip():
            out.append(0.0)
            continue
        sim = edit_similarity(pred, gt)
        short = len(pred) < UNDER_EXTRACTION_LEN * len(gt)
        out.append(-1.0 if (sim < UNDER_EXTRACTION_SIM and short) else 0.0)
    return out


def leak_penalty(completions: list[Any], **kwargs: Any) -> list[float]:
    del kwargs
    return [-1.0 if has_format_leak(completion_text(c)) else 0.0 for c in completions]


REWARD_FUNCS = [
    edit_reward,
    empty_reward,
    over_extraction_penalty,
    under_extraction_penalty,
    leak_penalty,
]

REWARD_WEIGHTS: list[float] = [
    DEFAULT_WEIGHTS.edit,
    DEFAULT_WEIGHTS.empty,
    DEFAULT_WEIGHTS.over_extraction,
    DEFAULT_WEIGHTS.under_extraction,
    DEFAULT_WEIGHTS.leak,
]


def _terms(completions: list[Any], target: Any, is_negative: Any) -> dict[str, list[float]]:
    return {
        "edit": edit_reward(completions, target, is_negative),
        "empty": empty_reward(completions, target, is_negative),
        "over_extraction": over_extraction_penalty(completions, target, is_negative),
        "under_extraction": under_extraction_penalty(completions, target, is_negative),
        "leak": leak_penalty(completions),
    }


def weighted_rewards(
    completions: list[Any],
    target: Any,
    is_negative: Any = None,
    *,
    weights: Q1RewardWeights = DEFAULT_WEIGHTS,
) -> list[float]:
    t = _terms(completions, target, is_negative)
    w = weights
    return [
        w.edit * t["edit"][i]
        + w.empty * t["empty"][i]
        + w.over_extraction * t["over_extraction"][i]
        + w.under_extraction * t["under_extraction"][i]
        + w.leak * t["leak"][i]
        for i in range(len(completions))
    ]


def reward_breakdown(
    completions: list[Any],
    target: Any,
    is_negative: Any = None,
    *,
    weights: Q1RewardWeights = DEFAULT_WEIGHTS,
) -> list[dict[str, float]]:
    t = _terms(completions, target, is_negative)
    total = weighted_rewards(completions, target, is_negative, weights=weights)
    rows: list[dict[str, float]] = []
    for i in range(len(completions)):
        row = {k: v[i] for k, v in t.items()}
        row["total"] = total[i]
        rows.append(row)
    return rows

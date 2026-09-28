"""GRPO rewards for Q2 (point OCR + English→Chinese).

Why a Q2-specific reward set
----------------------------
The Q2 SFT eval (``eval/run_q2_eval.py``) separated three axes and showed:

* **source / localization is already good** — 89.3% source hit, 0.931 mean edit
  similarity.  GRPO should *not* burn its budget re-learning localization.
* **empty discipline is the weak axis** — only 65.8% effective-empty on
  negatives; 34.2% hallucinate a block, 11.7% emit a broken/truncated shell.
* **translation cannot be scored with edit distance** — the reference is one
  OPUS wording; a faithful paraphrase scores ~0.4.  chrF is a better-behaved
  (though still reference-based) proxy.
* **structure can break** — 1.5% of positives never close ``</translation>``.

So the reward is a linear combination of:

======================  ======  ===========================================
component               weight  what it shapes
======================  ======  ===========================================
``source_reward``        0.5    keep the OCR block correct (do not regress)
``translation_reward``   1.0    chrF2 vs reference translation
``xml_reward``           0.5    emit a well-formed ``<source>/<translation>``
``empty_reward``         1.5    stay empty on negatives; never empty on positives
``hallucination_penalty``1.5    heavy penalty for inventing a block on empty
``over_extraction_penalty``0.5  penalty for dumping neighbours / the page
``leak_penalty``         0.5    chat/think control-token leakage
======================  ======  ===========================================

Design notes
------------
* **Negatives dominate the gradient on purpose.**  ``empty_reward`` +
  ``hallucination_penalty`` is the largest term, because that is the axis that
  actually failed.  Strict empty gets full credit; a shell/bare ``<source>``
  gets half (it is not a hallucination, but it is not clean either).
* **Source and translation are gated, not independent.**  A positive that
  outputs no source cannot earn translation credit (there is nothing to
  translate).  This prevents reward hacking by emitting a plausible Chinese
  sentence unrelated to the marked block.
* **chrF, not edit distance.**  Edit distance punished valid paraphrase and
  made a "good" model look like 0.43.  chrF2 is cheap and reference-based; swap
  in COMET later if available.
* **Penalties are separate terms** (not folded into a single hit flag) so a
  GRPO group can still rank "close" rollouts against each other.
* The reward is **dense** (edit/chrF in [0,1]) so a group of G=8 samples gets
  a usable advantage signal instead of an all-or-nothing hit.

This module is a *prototype*: weights are deliberately explicit and tunable,
and :func:`reward_breakdown` exposes each term for inspection before training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from point_ocr.grpo_rewards import cleaned_pred, completion_text
from point_ocr.infer import has_format_leak
from point_ocr.metrics import edit_similarity, is_empty_pred, looks_like_over_extraction
from point_ocr.q2_metrics import chrf, parse_ocr_mt_prediction


@dataclass(frozen=True)
class Q2RewardWeights:
    source: float = 0.5
    translation: float = 1.0
    xml: float = 0.5
    empty: float = 1.5
    hallucination: float = 1.5
    over_extraction: float = 0.5
    leak: float = 0.5


DEFAULT_WEIGHTS = Q2RewardWeights()
# Reward order for TRL's `reward_weights` (must match REWARD_FUNCS below).
WEIGHT_FIELDS = (
    "source",
    "translation",
    "xml",
    "empty",
    "hallucination",
    "over_extraction",
    "leak",
)


def weights_from_env(default: Q2RewardWeights = DEFAULT_WEIGHTS) -> Q2RewardWeights:
    """Override the Q2 reward weights with `GRPO_REWARD_WEIGHTS="s,t,xml,emp,hal,ovr,leak"`.

    Needed to focus RL on one axis: e.g. localization-first training sets the
    translation weight near zero (``1.5,0.05,1.0,1.5,1.5,0.75,0.5``) so the
    gradient shapes `<source>` + XML structure, not chrF on a single reference.
    The composer reads the same env var so the signal gate uses the same scale.
    """
    import os

    raw = (os.environ.get("GRPO_REWARD_WEIGHTS") or "").strip()
    if not raw:
        return default
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if len(vals) != len(WEIGHT_FIELDS):
        raise ValueError(
            f"GRPO_REWARD_WEIGHTS needs {len(WEIGHT_FIELDS)} numbers "
            f"({','.join(WEIGHT_FIELDS)}), got {len(vals)}"
        )
    return Q2RewardWeights(*vals)


def positive_ceiling(weights: Q2RewardWeights = DEFAULT_WEIGHTS) -> float:
    """Max reward a *positive* row can earn (empty/hallucination do not apply)."""
    return float(weights.source + weights.translation + weights.xml)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _flags(is_negative: Any, n: int, targets: list[str]) -> list[bool]:
    if isinstance(is_negative, list):
        out = [_as_bool(v) for v in is_negative]
        if len(out) < n:
            out.extend([False] * (n - len(out)))
        return out[:n]
    if is_negative is not None:
        return [_as_bool(is_negative)] * n
    return [not t.strip() for t in targets]


def _targets(target: Any, n: int) -> list[str]:
    if isinstance(target, list):
        return [str(t or "") for t in target]
    return [str(target or "")] * n


def _paired(completions: list[Any], target: Any, is_negative: Any):
    n = len(completions)
    targets = _targets(target, n)
    flags = _flags(is_negative, n, targets)
    for completion, gt, neg in zip(completions, targets, flags):
        pred = cleaned_pred(completion)
        yield pred, gt, neg


# --------------------------------------------------------------------------- #
# Individual reward terms (TRL signature: (completions, **columns) -> list[float])
# --------------------------------------------------------------------------- #
def source_reward(completions: list[Any], target: Any, is_negative: Any = None, **kwargs: Any) -> list[float]:
    """Edit similarity of the parsed ``<source>`` to the GT source. 0 on negatives."""
    del kwargs
    out: list[float] = []
    for pred, gt, neg in _paired(completions, target, is_negative):
        if neg:
            out.append(0.0)
            continue
        p = parse_ocr_mt_prediction(pred)
        g = parse_ocr_mt_prediction(gt)
        out.append(float(edit_similarity(p.source, g.source)))
    return out


def translation_reward(completions: list[Any], target: Any, is_negative: Any = None, **kwargs: Any) -> list[float]:
    """chrF2 of the parsed ``<translation>``. Gated: no source -> no credit."""
    del kwargs
    out: list[float] = []
    for pred, gt, neg in _paired(completions, target, is_negative):
        if neg:
            out.append(0.0)
            continue
        p = parse_ocr_mt_prediction(pred)
        g = parse_ocr_mt_prediction(gt)
        if not p.source.strip():
            out.append(0.0)
            continue
        out.append(float(chrf(p.translation, g.translation, char_order=6, word_order=0)))
    return out


def xml_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Structure: +1 well-formed, +0.4 both tags but stray/unclosed, 0 one tag, -0.5 no XML."""
    del kwargs
    out: list[float] = []
    for completion in completions:
        pred = cleaned_pred(completion)
        if not pred.strip():
            out.append(0.0)
            continue
        p = parse_ocr_mt_prediction(pred)
        if p.well_formed:
            out.append(1.0)
        elif p.has_source and p.has_translation:
            out.append(0.4)
        elif p.has_source or p.has_translation:
            out.append(0.0)
        else:
            out.append(-0.5)  # plain text where XML was required
    return out


def empty_reward(completions: list[Any], target: Any, is_negative: Any = None, **kwargs: Any) -> list[float]:
    """Negatives: +1 strict empty, +0.5 broken-empty, 0 hallucinated. Positives: -1 if empty."""
    del kwargs
    out: list[float] = []
    for pred, _gt, neg in _paired(completions, target, is_negative):
        p = parse_ocr_mt_prediction(pred)
        strict_empty = is_empty_pred(pred)
        effective_empty = strict_empty or p.empty_shell or p.truncated
        if neg:
            if strict_empty:
                out.append(1.0)
            elif effective_empty:
                out.append(0.5)
            else:
                out.append(0.0)
        else:
            out.append(-1.0 if strict_empty else 0.0)
    return out


def hallucination_penalty(completions: list[Any], target: Any, is_negative: Any = None, **kwargs: Any) -> list[float]:
    """Negative GT but a non-empty source was produced -> -1."""
    del kwargs
    out: list[float] = []
    for pred, _gt, neg in _paired(completions, target, is_negative):
        if neg and parse_ocr_mt_prediction(pred).source.strip():
            out.append(-1.0)
        else:
            out.append(0.0)
    return out


def over_extraction_penalty(completions: list[Any], target: Any, is_negative: Any = None, **kwargs: Any) -> list[float]:
    """Positive GT but the predicted source dumps neighbours / the page -> -1."""
    del kwargs
    out: list[float] = []
    for pred, gt, neg in _paired(completions, target, is_negative):
        if neg:
            out.append(0.0)
            continue
        p = parse_ocr_mt_prediction(pred)
        g = parse_ocr_mt_prediction(gt)
        out.append(-1.0 if looks_like_over_extraction(p.source, g.source) else 0.0)
    return out


def leak_penalty(completions: list[Any], **kwargs: Any) -> list[float]:
    """Chat / thinking control tokens in the raw decode -> -1."""
    del kwargs
    return [-1.0 if has_format_leak(completion_text(c)) else 0.0 for c in completions]


REWARD_FUNCS = [
    source_reward,
    translation_reward,
    xml_reward,
    empty_reward,
    hallucination_penalty,
    over_extraction_penalty,
    leak_penalty,
]

# Linear weights matching REWARD_FUNCS order (used by TRL's reward_weights).
REWARD_WEIGHTS: list[float] = [
    DEFAULT_WEIGHTS.source,
    DEFAULT_WEIGHTS.translation,
    DEFAULT_WEIGHTS.xml,
    DEFAULT_WEIGHTS.empty,
    DEFAULT_WEIGHTS.hallucination,
    DEFAULT_WEIGHTS.over_extraction,
    DEFAULT_WEIGHTS.leak,
]


def weighted_rewards_q2(
    completions: list[Any],
    target: Any,
    is_negative: Any = None,
    *,
    weights: Q2RewardWeights | None = None,
) -> list[float]:
    """Linear combination matching TRL's weighted reward aggregation.

    `weights=None` resolves `GRPO_REWARD_WEIGHTS` so the value probe, the
    composer gate and the GRPO trainer all use the *same* scale.
    """
    weights = weights if weights is not None else weights_from_env()
    terms = {
        "source": source_reward(completions, target, is_negative),
        "translation": translation_reward(completions, target, is_negative),
        "xml": xml_reward(completions),
        "empty": empty_reward(completions, target, is_negative),
        "hallucination": hallucination_penalty(completions, target, is_negative),
        "over_extraction": over_extraction_penalty(completions, target, is_negative),
        "leak": leak_penalty(completions),
    }
    w = weights
    return [
        w.source * terms["source"][i]
        + w.translation * terms["translation"][i]
        + w.xml * terms["xml"][i]
        + w.empty * terms["empty"][i]
        + w.hallucination * terms["hallucination"][i]
        + w.over_extraction * terms["over_extraction"][i]
        + w.leak * terms["leak"][i]
        for i in range(len(completions))
    ]


def reward_breakdown(
    completions: list[Any],
    target: Any,
    is_negative: Any = None,
    *,
    weights: Q2RewardWeights | None = None,
) -> list[dict[str, float]]:
    """Per-completion term values + weighted total (for debugging GRPO groups)."""
    weights = weights if weights is not None else weights_from_env()
    terms = {
        "source": source_reward(completions, target, is_negative),
        "translation": translation_reward(completions, target, is_negative),
        "xml": xml_reward(completions),
        "empty": empty_reward(completions, target, is_negative),
        "hallucination": hallucination_penalty(completions, target, is_negative),
        "over_extraction": over_extraction_penalty(completions, target, is_negative),
        "leak": leak_penalty(completions),
    }
    total = weighted_rewards_q2(completions, target, is_negative, weights=weights)
    rows: list[dict[str, float]] = []
    for i in range(len(completions)):
        row = {k: v[i] for k, v in terms.items()}
        row["total"] = total[i]
        rows.append(row)
    return rows

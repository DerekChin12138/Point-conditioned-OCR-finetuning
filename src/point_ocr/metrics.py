"""Evaluation metrics for point-conditioned OCR.

Primary metrics (held-out):
  - block_hit_rate: prediction matches GT block (normalized edit similarity)
  - over_extraction_rate: prediction looks like full-page dump vs short block
  - empty_on_chrome_rate: empty output when GT is empty (negative / chrome)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import editdistance


_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _WS.sub(" ", s)
    return s


def edit_similarity(a: str, b: str) -> float:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    dist = editdistance.eval(a_n, b_n)
    return 1.0 - dist / max(len(a_n), len(b_n))


def normalized_edit_distance(a: str, b: str) -> float:
    """1 - edit_similarity; 0 = identical after normalize."""
    return 1.0 - edit_similarity(a, b)


def is_empty_pred(pred: str) -> bool:
    return normalize_text(pred) == ""


def looks_like_over_extraction(
    pred: str,
    gt: str,
    *,
    min_pred_chars: int = 400,
    ratio_vs_gt: float = 3.0,
    multi_heading: bool = True,
) -> bool:
    """Heuristic: prediction much longer than GT block / multi-section dump."""
    p = pred.strip()
    g = gt.strip()
    if not p:
        return False
    if len(p) >= min_pred_chars and (not g or len(p) >= ratio_vs_gt * max(len(g), 1)):
        return True
    if multi_heading and g and len(p) > max(200, 2 * len(g)):
        headings = len(re.findall(r"(?m)^#{1,6}\s", p))
        if headings >= 3:
            return True
        # Many blank-line separated blocks
        blocks = [b for b in re.split(r"\n\s*\n", p) if b.strip()]
        if len(blocks) >= 5 and len(p) > 2 * max(len(g), 1):
            return True
    return False


@dataclass
class EvalExample:
    sample_id: str
    prediction: str
    target: str
    is_negative: bool = False


@dataclass
class MetricReport:
    n: int
    block_hit_rate: float
    over_extraction_rate: float
    empty_on_chrome_rate: float
    mean_edit_similarity: float
    mean_normalized_edit_distance: float
    n_positive: int
    n_negative: int

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "block_hit_rate": self.block_hit_rate,
            "over_extraction_rate": self.over_extraction_rate,
            "empty_on_chrome_rate": self.empty_on_chrome_rate,
            "mean_edit_similarity": self.mean_edit_similarity,
            "mean_normalized_edit_distance": self.mean_normalized_edit_distance,
        }


def evaluate_examples(
    examples: list[EvalExample],
    *,
    hit_threshold: float = 0.85,
) -> MetricReport:
    if not examples:
        return MetricReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    sims: list[float] = []
    dists: list[float] = []
    hits = 0
    over = 0
    pos = 0
    neg = 0
    empty_ok = 0

    for ex in examples:
        sim = edit_similarity(ex.prediction, ex.target)
        sims.append(sim)
        dists.append(normalized_edit_distance(ex.prediction, ex.target))
        if ex.is_negative or is_empty_pred(ex.target):
            neg += 1
            if is_empty_pred(ex.prediction):
                empty_ok += 1
            if looks_like_over_extraction(ex.prediction, ex.target):
                over += 1
        else:
            pos += 1
            if sim >= hit_threshold:
                hits += 1
            if looks_like_over_extraction(ex.prediction, ex.target):
                over += 1

    n = len(examples)
    return MetricReport(
        n=n,
        block_hit_rate=(hits / pos) if pos else 0.0,
        over_extraction_rate=over / n,
        empty_on_chrome_rate=(empty_ok / neg) if neg else 0.0,
        mean_edit_similarity=sum(sims) / n,
        mean_normalized_edit_distance=sum(dists) / n,
        n_positive=pos,
        n_negative=neg,
    )


def evaluate_by_bucket(
    examples: list[EvalExample],
    bucket_keys: list[str],
    *,
    hit_threshold: float = 0.85,
) -> dict[str, MetricReport]:
    """Group examples by parallel bucket labels and score each group."""
    if len(examples) != len(bucket_keys):
        raise ValueError("examples and bucket_keys must have the same length")
    groups: dict[str, list[EvalExample]] = {}
    for ex, key in zip(examples, bucket_keys):
        groups.setdefault(key or "unknown", []).append(ex)
    return {
        key: evaluate_examples(group, hit_threshold=hit_threshold)
        for key, group in sorted(groups.items())
    }

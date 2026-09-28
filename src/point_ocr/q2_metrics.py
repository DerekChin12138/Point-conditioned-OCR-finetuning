"""Q2 (point OCR + English→Chinese) evaluation metrics.

The legacy whole-string metric conflated three different axes:

1. **localization / OCR** of the marked block  -> ``source_*``
2. **translation** quality                    -> ``translation_*`` (chrF, not edit distance)
3. **empty-output discipline**                -> ``empty_*`` / ``hallucination_*``

This module scores a Q2 prediction JSONL (the same rows written by
``train/run_observability.run_final_generate_eval``) without needing a GPU.

Prediction / target contract (``prompt_key=ocr_mt_v1``)::

    <source>
    original Markdown
    </source>
    <translation>
    Simplified Chinese
    </translation>

Empty GT (blank / chrome / table / image) stays ``""``.  A prediction that is
only a bare ``<source>`` token, or an XML shell with empty tags, is *not* a
hallucination but is also not a clean empty answer — it is counted separately.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from point_ocr.infer import has_format_leak
from point_ocr.metrics import edit_similarity, is_empty_pred, looks_like_over_extraction

_SOURCE_RE = re.compile(r"<source>\s*(.*?)\s*</source>", re.IGNORECASE | re.DOTALL)
_TRANS_RE = re.compile(r"<translation>\s*(.*?)\s*</translation>", re.IGNORECASE | re.DOTALL)
_ANY_TAG_RE = re.compile(r"</?(?:source|translation)>", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OcrMtParse:
    source: str = ""
    translation: str = ""
    has_source: bool = False
    has_translation: bool = False
    open_source: bool = False
    unclosed_source: bool = False
    unclosed_translation: bool = False
    source_before_translation: bool = False
    stray: str = ""

    @property
    def well_formed(self) -> bool:
        """Both tags present, source first, nothing outside the two elements."""
        return (
            self.has_source
            and self.has_translation
            and self.source_before_translation
            and not self.stray.strip()
        )

    @property
    def empty_shell(self) -> bool:
        """``<source></source><translation></translation>`` — nothing inside."""
        return self.has_source and self.has_translation and not self.source.strip() and not self.translation.strip()

    @property
    def truncated(self) -> bool:
        """Generation stopped right after an opening tag (e.g. bare ``<source>``)."""
        return self.open_source and not self.has_source and not self.has_translation


def parse_ocr_mt_prediction(text: str) -> OcrMtParse:
    raw = text or ""
    sm = _SOURCE_RE.search(raw)
    tm = _TRANS_RE.search(raw)
    stripped = _TRANS_RE.sub("", _SOURCE_RE.sub("", raw))
    has_source = sm is not None
    has_translation = tm is not None
    open_source = bool(re.search(r"<source>", raw, re.IGNORECASE))
    return OcrMtParse(
        source=(sm.group(1) if sm else ""),
        translation=(tm.group(1) if tm else ""),
        has_source=has_source,
        has_translation=has_translation,
        open_source=open_source,
        unclosed_source=open_source and "</source>" not in raw.lower(),
        unclosed_translation=(
            "<translation>" in raw.lower() and "</translation>" not in raw.lower()
        ),
        source_before_translation=bool(
            has_source and has_translation and sm.start() < tm.start()
        ),
        stray=stripped,
    )


# --------------------------------------------------------------------------- #
# chrF (character n-gram F-score; chrF2 default, chrF++ with word_order=2)
# --------------------------------------------------------------------------- #
def _ngrams(seq: Sequence[Any], n: int) -> Counter:
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def _f_score(hyp_ng: Counter, ref_ng: Counter, beta: float) -> float:
    matches = sum((hyp_ng & ref_ng).values())
    h = sum(hyp_ng.values())
    r = sum(ref_ng.values())
    if h == 0 or r == 0:
        return 0.0
    precision = matches / h
    recall = matches / r
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall)


def chrf(
    hypothesis: str,
    reference: str,
    *,
    char_order: int = 6,
    word_order: int = 0,
    beta: float = 2.0,
) -> float:
    """chrF-beta between two strings (whitespace-normalized).

    ``word_order=2`` gives chrF++.  Empty hypothesis/empty reference = 1.0 only
    when both are empty, else 0.0.
    """
    hyp = " ".join((hypothesis or "").split())
    ref = " ".join((reference or "").split())
    if not hyp or not ref:
        return 1.0 if hyp == ref else 0.0

    scores: list[float] = []
    max_char = min(char_order, len(hyp), len(ref))
    for n in range(1, max_char + 1):
        scores.append(_f_score(_ngrams(hyp, n), _ngrams(ref, n), beta))
    if word_order > 0:
        hw, rw = hyp.split(), ref.split()
        max_word = min(word_order, len(hw), len(rw))
        for n in range(1, max_word + 1):
            scores.append(_f_score(_ngrams(hw, n), _ngrams(rw, n), beta))
    return sum(scores) / len(scores) if scores else 0.0


# --------------------------------------------------------------------------- #
# Examples / report
# --------------------------------------------------------------------------- #
@dataclass
class Q2Example:
    sample_id: str
    prediction: str
    target: str
    is_negative: bool = False
    raw_prediction: str | None = None
    bucket: str = ""


@dataclass
class Q2MetricReport:
    n: int = 0
    n_positive: int = 0
    n_negative: int = 0
    # positives
    xml_pair_rate: float = 0.0
    xml_well_formed_rate: float = 0.0
    source_hit_rate: float = 0.0
    source_edit_similarity: float = 0.0
    translation_chrf: float = 0.0
    translation_chrfpp: float = 0.0
    translation_hit_rate: float = 0.0
    both_hit_rate: float = 0.0
    translation_empty_rate: float = 0.0
    translation_cjk_ratio: float = 0.0
    translation_length_ratio: float = 0.0
    unclosed_translation_rate: float = 0.0
    positive_over_extraction_rate: float = 0.0
    # negatives
    strict_empty_rate: float = 0.0
    effective_empty_rate: float = 0.0
    broken_empty_rate: float = 0.0
    hallucination_rate: float = 0.0
    plain_text_rate: float = 0.0
    negative_over_extraction_rate: float = 0.0
    # misc
    format_leak_rate: float = 0.0
    # thresholds used
    source_threshold: float = 0.85
    translation_threshold: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _translation_metrics(pred_tr: str, gt_tr: str) -> tuple[float, float, float, float, float]:
    """(chrf, chrfpp, cjk_ratio, length_ratio, empty_flag)."""
    p = (pred_tr or "").strip()
    g = (gt_tr or "").strip()
    c = chrf(p, g, char_order=6, word_order=0)
    cp = chrf(p, g, char_order=6, word_order=2)
    cjk = len(_CJK_RE.findall(p)) / len(p) if p else 0.0
    ratio = len(p) / len(g) if g else (1.0 if not p else 0.0)
    return c, cp, cjk, ratio, (1.0 if not p else 0.0)


def evaluate_q2(
    examples: Sequence[Q2Example],
    *,
    source_threshold: float = 0.85,
    translation_threshold: float = 0.60,
) -> Q2MetricReport:
    if not examples:
        return Q2MetricReport(source_threshold=source_threshold, translation_threshold=translation_threshold)

    pos_src_hit = pos_pair = pos_well = pos_both = pos_tr_hit = 0
    src_sims: list[float] = []
    chrfs: list[float] = []
    chrfpps: list[float] = []
    cjks: list[float] = []
    ratios: list[float] = []
    tr_empty = unclosed = pos_over = 0
    neg = strict = effective = broken = halluc = plain = neg_over = 0
    leaks = 0

    for ex in examples:
        pred = ex.prediction or ""
        gt = ex.target or ""
        raw = ex.raw_prediction if ex.raw_prediction is not None else pred
        if has_format_leak(raw):
            leaks += 1

        p = parse_ocr_mt_prediction(pred)
        g = parse_ocr_mt_prediction(gt)

        if ex.is_negative or is_empty_pred(gt):
            neg += 1
            if not pred.strip():
                strict += 1
            if (not pred.strip()) or p.empty_shell or p.truncated:
                effective += 1
                if pred.strip():
                    broken += 1
            if p.source.strip():
                halluc += 1
                if looks_like_over_extraction(p.source, ""):
                    neg_over += 1
            elif pred.strip() and not p.has_source and not p.has_translation:
                plain += 1
            continue

        # positive
        pos_src_hit_flag = False
        if p.has_source:
            pos_pair += 1
        if p.well_formed:
            pos_well += 1
        sim = edit_similarity(p.source, g.source)
        src_sims.append(sim)
        if sim >= source_threshold:
            pos_src_hit += 1
            pos_src_hit_flag = True
        c, cp, cjk, ratio, tr_emp = _translation_metrics(p.translation, g.translation)
        chrfs.append(c)
        chrfpps.append(cp)
        cjks.append(cjk)
        ratios.append(ratio)
        if tr_emp:
            tr_empty += 1
        if c >= translation_threshold:
            pos_tr_hit += 1
        if pos_src_hit_flag and c >= translation_threshold:
            pos_both += 1
        if p.unclosed_translation:
            unclosed += 1
        if looks_like_over_extraction(p.source, g.source):
            pos_over += 1

    n_pos = len(src_sims)
    n = len(examples)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return Q2MetricReport(
        n=n,
        n_positive=n_pos,
        n_negative=neg,
        xml_pair_rate=(pos_pair / n_pos) if n_pos else 0.0,
        xml_well_formed_rate=(pos_well / n_pos) if n_pos else 0.0,
        source_hit_rate=(pos_src_hit / n_pos) if n_pos else 0.0,
        source_edit_similarity=mean(src_sims),
        translation_chrf=mean(chrfs),
        translation_chrfpp=mean(chrfpps),
        translation_hit_rate=(pos_tr_hit / n_pos) if n_pos else 0.0,
        both_hit_rate=(pos_both / n_pos) if n_pos else 0.0,
        translation_empty_rate=(tr_empty / n_pos) if n_pos else 0.0,
        translation_cjk_ratio=mean(cjks),
        translation_length_ratio=mean(ratios),
        unclosed_translation_rate=(unclosed / n_pos) if n_pos else 0.0,
        positive_over_extraction_rate=(pos_over / n_pos) if n_pos else 0.0,
        strict_empty_rate=(strict / neg) if neg else 0.0,
        effective_empty_rate=(effective / neg) if neg else 0.0,
        broken_empty_rate=(broken / neg) if neg else 0.0,
        hallucination_rate=(halluc / neg) if neg else 0.0,
        plain_text_rate=(plain / neg) if neg else 0.0,
        negative_over_extraction_rate=(neg_over / neg) if neg else 0.0,
        format_leak_rate=(leaks / n) if n else 0.0,
        source_threshold=source_threshold,
        translation_threshold=translation_threshold,
    )


def evaluate_q2_by_bucket(
    examples: Sequence[Q2Example],
    bucket_keys: Sequence[str],
    *,
    source_threshold: float = 0.85,
    translation_threshold: float = 0.60,
) -> dict[str, Q2MetricReport]:
    if len(examples) != len(bucket_keys):
        raise ValueError("examples and bucket_keys must have the same length")
    groups: dict[str, list[Q2Example]] = {}
    for ex, key in zip(examples, bucket_keys):
        groups.setdefault(key or "unknown", []).append(ex)
    return {
        key: evaluate_q2(group, source_threshold=source_threshold, translation_threshold=translation_threshold)
        for key, group in sorted(groups.items())
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def report_to_markdown(report: dict[str, Any]) -> str:
    """Render the dict from :func:`evaluate_q2` (+ optional buckets) as Markdown."""
    o = report.get("overall", report)

    def pct(key: str, d: dict[str, Any] | None = None) -> str:
        d = d if d is not None else o
        return f"{100.0 * float(d.get(key, 0.0)):.1f}%"

    def num(key: str, d: dict[str, Any] | None = None) -> str:
        d = d if d is not None else o
        return f"{float(d.get(key, 0.0)):.3f}"

    lines = [
        "# Q2 (OCR + EN→ZH) evaluation",
        "",
        f"- scored rows: **{o.get('n', 0)}** (positive {o.get('n_positive', 0)} / negative {o.get('n_negative', 0)})",
        f"- thresholds: source ≥ {o.get('source_threshold')} edit-sim, translation ≥ {o.get('translation_threshold')} chrF",
        "",
        "## Positives — localization",
        "| metric | value |",
        "|---|---|",
        f"| XML pair rate (has `<source>`) | {pct('xml_pair_rate')} |",
        f"| XML well-formed rate | {pct('xml_well_formed_rate')} |",
        f"| source hit rate | {pct('source_hit_rate')} |",
        f"| source edit similarity (mean) | {num('source_edit_similarity')} |",
        f"| positive over-extraction | {pct('positive_over_extraction_rate')} |",
        "",
        "## Positives — translation",
        "| metric | value |",
        "|---|---|",
        f"| chrF2 (mean) | {num('translation_chrf')} |",
        f"| chrF++ (mean) | {num('translation_chrfpp')} |",
        f"| translation hit rate | {pct('translation_hit_rate')} |",
        f"| both (source ∧ translation) hit | {pct('both_hit_rate')} |",
        f"| empty translation | {pct('translation_empty_rate')} |",
        f"| CJK char ratio (mean) | {num('translation_cjk_ratio')} |",
        f"| length ratio pred/GT (mean) | {num('translation_length_ratio')} |",
        f"| unclosed `</translation>` | {pct('unclosed_translation_rate')} |",
        "",
        "## Negatives — empty discipline",
        "| metric | value |",
        "|---|---|",
        f"| strict empty (exactly `\"\"`) | {pct('strict_empty_rate')} |",
        f"| effective empty (+shell/bare `<source>`) | {pct('effective_empty_rate')} |",
        f"| broken empty (shell/truncated, not clean) | {pct('broken_empty_rate')} |",
        f"| hallucination (non-empty source) | {pct('hallucination_rate')} |",
        f"| plain text (no XML) | {pct('plain_text_rate')} |",
        f"| negative over-extraction | {pct('negative_over_extraction_rate')} |",
        "",
        f"format leak: {pct('format_leak_rate')}",
    ]
    buckets = report.get("by_bucket") or {}
    if buckets:
        lines += ["", "## By bucket", "", "| bucket | n | src hit | chrF2 | both hit | eff-empty | halluc |", "|---|---|---|---|---|---|---|"]
        for key, d in buckets.items():
            lines.append(
                f"| {key} | {d.get('n', 0)} | {pct('source_hit_rate', d)} | {num('translation_chrf', d)} "
                f"| {pct('both_hit_rate', d)} | {pct('effective_empty_rate', d)} | {pct('hallucination_rate', d)} |"
            )
    return "\n".join(lines) + "\n"

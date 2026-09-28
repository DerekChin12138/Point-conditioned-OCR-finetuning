"""Tests for A2 mix quotas and interleaving."""

from __future__ import annotations

import random

from point_ocr.a2_mix import (
    A2_SLICE_FRAC,
    A2_TARGET_DEFAULT,
    interleave_a2_samples,
    interleave_by_key,
    quota_counts,
    template_stem_from_page_id,
)


def test_quota_counts_sum_and_approx_frac():
    q = quota_counts(A2_TARGET_DEFAULT)
    assert sum(q.values()) == A2_TARGET_DEFAULT
    assert set(q) == set(A2_SLICE_FRAC)
    for k, frac in A2_SLICE_FRAC.items():
        assert abs(q[k] / A2_TARGET_DEFAULT - frac) < 0.01


def test_template_stem_from_page_id():
    assert template_stem_from_page_id("desk__23_ide_dense_split") == "23_ide_dense_split"
    assert template_stem_from_page_id("replay__01_article_twocol__0") == "01_article_twocol"
    assert template_stem_from_page_id("semgroup__27") == "27"


def test_interleave_breaks_long_runs():
    # Clustered by template then interleaved
    items = [f"t{i % 5}_{j}" for i in range(5) for j in range(40)]
    rng = random.Random(0)
    out = interleave_by_key(items, key_fn=lambda x: x.split("_")[0], rng=rng)
    assert len(out) == len(items)
    assert set(out) == set(items)
    max_run = 1
    run = 1
    for a, b in zip(out, out[1:]):
        if a.split("_")[0] == b.split("_")[0]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    assert max_run <= 3  # round-robin across 5 equal buckets → usually 1


def test_interleave_a2_samples_slice_template():
    samples = []
    for sl in ("replay", "desktop", "adjacency"):
        for tm in ("01_article", "22_messy", "25_win"):
            for i in range(10):
                samples.append({"slice": sl, "tmpl": tm, "i": i})
    # clustered order
    samples.sort(key=lambda s: (s["slice"], s["tmpl"]))
    out = interleave_a2_samples(
        samples,
        rng=random.Random(1),
        slice_fn=lambda s: s["slice"],
        template_fn=lambda s: s["tmpl"],
    )
    assert len(out) == len(samples)
    max_run = 1
    run = 1
    for a, b in zip(out, out[1:]):
        if (a["slice"], a["tmpl"]) == (b["slice"], b["tmpl"]):
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    assert max_run <= 4

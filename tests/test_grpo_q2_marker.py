"""Tests for GRPO_2 dual-size marker compose helpers."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from compose_grpo_q2_marker import assign_area_scales, resolve_unmarked  # noqa: E402
from point_ocr.marker import (  # noqa: E402
    MARKER_AREA_FRAC,
    MARKER_AREA_FRAC_LEGACY,
    MARKER_AREA_FRAC_MAX,
    MARKER_AREA_FRAC_MIN,
)


def test_assign_area_scales_hard_mix():
    scenes = ["multi_frag"] * 70 + ["semantic_group"] * 30
    scales = assign_area_scales(
        scenes,
        hard_scenes={"multi_frag", "semantic_group"},
        large=0.85,
        small=0.5,
        small_frac=0.7,
        rng=random.Random(0),
    )
    assert len(scales) == 100
    assert sum(1 for s in scales if abs(s - 0.5) < 1e-9) == 70
    assert sum(1 for s in scales if abs(s - 0.85) < 1e-9) == 30


def test_assign_area_scales_non_hard_all_large():
    scenes = ["regular", "empty", "regular"]
    scales = assign_area_scales(
        scenes,
        hard_scenes={"multi_frag", "semantic_group"},
        large=0.85,
        small=0.5,
        small_frac=0.7,
        rng=random.Random(1),
    )
    assert scales == [0.85, 0.85, 0.85]


def test_default_area_frac_is_range_midpoint():
    assert abs(MARKER_AREA_FRAC - (MARKER_AREA_FRAC_MIN + MARKER_AREA_FRAC_MAX) / 2) < 1e-12
    assert MARKER_AREA_FRAC_LEGACY == 0.005


def test_resolve_unmarked_synth_page():
    meta = {
        "pool_id": "multi_frag",
        "page_id": "multi_frag__14_magazine_3col__97",
    }
    path = resolve_unmarked(meta)
    assert path is not None
    assert path.is_file()

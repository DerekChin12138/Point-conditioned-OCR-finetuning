from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "marker_studio"))

from quad_probe import expected_color, interpret_probe


def test_quadrant_colors_top_left_origin():
    assert expected_color(25, 25, origin="top-left") == "red"
    assert expected_color(75, 25, origin="top-left") == "green"
    assert expected_color(25, 75, origin="top-left") == "blue"
    assert expected_color(75, 75, origin="top-left") == "yellow"


def test_quadrant_colors_bottom_left_origin():
    assert expected_color(25, 25, origin="bottom-left") == "blue"
    assert expected_color(25, 75, origin="bottom-left") == "red"


def test_interpret_top_left_prior():
    preds = [
        {"x_pct": 25, "y_pct": 25, "pred": "red"},
        {"x_pct": 25, "y_pct": 75, "pred": "blue"},
        {"x_pct": 75, "y_pct": 25, "pred": "green"},
        {"x_pct": 75, "y_pct": 75, "pred": "yellow"},
    ]
    assert "TOP-LEFT" in interpret_probe(preds)

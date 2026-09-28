"""GRPO TensorBoard/log_history → per-metric PNG helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from run_observability import GRPO_CURVE_METRICS, series_from_log_history


def test_series_skips_nan_and_missing():
    logs = [
        {"step": 5, "reward": 0.5, "entropy": float("nan")},
        {"step": 10, "reward": 0.7, "kl": 0.01},
        {"loss": 0.1},
    ]
    steps, vals = series_from_log_history(logs, "reward")
    assert steps == [5, 10]
    assert vals == [0.5, 0.7]
    e_steps, e_vals = series_from_log_history(logs, "entropy")
    assert e_steps == []
    assert e_vals == []
    k_steps, k_vals = series_from_log_history(logs, "kl")
    assert k_steps == [10]
    assert k_vals == [0.01]


def test_selected_curve_keys():
    keys = [k for k, _ in GRPO_CURVE_METRICS]
    assert keys == [
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "kl",
        "entropy",
        "clip_ratio/region_mean",
        "grad_norm",
        "learning_rate",
    ]
    assert math.isnan(float("nan"))

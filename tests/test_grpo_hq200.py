"""GRPO HQ-200 data pipeline: candidate builder + signal-gated composer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


builder = _load("build_grpo_candidates", "data/scripts/build_grpo_candidates.py")
composer = _load("compose_grpo_q1_hq200", "data/scripts/compose_grpo_q1_hq200.py")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _row(pool: str, gt: str, *, page: str = "p1", tmpl: str = "t1", area: float = 0.0014,
         n_frag: int = 1, chrome: str = "none", sid: str | None = None) -> dict:
    return {
        "images": [f"/tmp/{sid or page}.jpg"],
        "messages": [
            {"role": "user", "content": "<image>find the block"},
            {"role": "assistant", "content": gt},
        ],
        "metadata": {
            "sample_id": sid or f"{pool}:{page}:{gt[:8]}",
            "pool_id": pool,
            "page_id": page,
            "template_stem": tmpl,
            "marker_area_frac": area,
            "n_fragments": n_frag,
            "is_negative": False,
            "chrome_family": chrome,
        },
    }


# --------------------------------------------------------------------------- #
# builder: classification / scoring
# --------------------------------------------------------------------------- #
def test_reason_classification():
    assert builder.reason_of(_row("multi_frag", "a" * 50), long_threshold=100) == "multi_frag"
    assert builder.reason_of(_row("semantic_group", "a" * 50), long_threshold=100) == "semantic_group"
    assert builder.reason_of(_row("real_labeled", "a" * 50), long_threshold=100) == "real"
    long_gt = " ".join(f"w{i}" for i in range(80))
    assert builder.reason_of(_row("core_inner", long_gt), long_threshold=100) == "long_block"
    assert builder.reason_of(_row("core_inner", "short varied block"), long_threshold=100) is None
    assert builder.reason_of(_row("core_inner", "x", n_frag=2), long_threshold=100) == "multi_frag"
    assert builder.reason_of(_row("core_inner", ""), long_threshold=100) is None


def test_repeat_risk_detection():
    assert builder._repeat_risk("♫ la la ♫ la la")
    assert builder._repeat_risk("same line\nsame line")
    assert builder._repeat_risk("| a | b | c | d | e | f |")
    assert builder._repeat_risk("word " * 40)
    assert not builder._repeat_risk("A normal paragraph with varied wording and no repeats.")


def test_hard_score_prefers_real_and_small_markers():
    real = builder.hard_score(_row("real_labeled", "x" * 200, chrome="chrome"), "real", tertile=0)
    long_ = builder.hard_score(_row("core_inner", "x" * 200), "long_block", tertile=2)
    assert real > long_
    small = builder.hard_score(_row("real_labeled", "x"), "real", tertile=0)
    big = builder.hard_score(_row("real_labeled", "x"), "real", tertile=2)
    assert small > big


# --------------------------------------------------------------------------- #
# builder: diversity guards
# --------------------------------------------------------------------------- #
def test_dhash_and_hamming(tmp_path: Path):
    from PIL import Image

    a = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(a)
    h1 = builder.crop_dhash(str(a), None)
    h2 = builder.crop_dhash(str(a), None)
    assert h1 is not None and h1 == h2
    assert builder._hamming(h1, h2) == 0
    b = tmp_path / "b.jpg"
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    for x in range(32):
        for y in range(64):
            img.putpixel((x, y), (255, 255, 255))
    img.save(b)
    h3 = builder.crop_dhash(str(b), None)
    assert builder._hamming(h1, h3) > 0


def test_build_pool_respects_page_cap_and_tertiles(tmp_path: Path):
    from PIL import Image

    # 3 pages x 6 rows, two templates, three marker sizes
    rows = []
    for p in range(3):
        for i in range(6):
            img = tmp_path / f"{p}_{i}.jpg"
            Image.new("RGB", (32, 32), (p * 40, i * 20, 10)).save(img)
            r = _row("multi_frag", "x" * 60, page=f"page{p}", tmpl=f"t{p % 2}",
                     area=[0.0010, 0.0014, 0.0018][i % 3], sid=f"s{p}_{i}")
            r["images"] = [str(img)]
            rows.append(r)
    out, stats = builder.build_pool(
        rows, target=6, mix={"multi_frag": 1.0}, seed=0, dedup=False
    )
    assert len(out) == 6
    pages = [r["metadata"]["page_id"] for r in out]
    from collections import Counter

    assert max(Counter(pages).values()) <= builder.PAGE_CAP_MAX
    tertiles = Counter(r["metadata"]["grp_tertile"] for r in out)
    assert max(tertiles.values()) <= 3  # balanced inside the bucket
    # the requested mix is written into split_meta-style stats
    assert stats["picked_by_reason"]["multi_frag"] == 6


def test_build_pool_dedups_near_duplicate_crops(tmp_path: Path):
    from PIL import Image

    img = tmp_path / "same.jpg"
    Image.new("RGB", (64, 64), (200, 100, 50)).save(img)
    rows = [_row("multi_frag", "x" * 60, page=f"pg{i}", sid=f"d{i}") for i in range(5)]
    for r in rows:
        r["images"] = [str(img)]
    out, _ = builder.build_pool(rows, target=5, mix={"multi_frag": 1.0}, seed=1, dedup=True)
    assert len(out) == 1  # all identical crops -> one survivor


# --------------------------------------------------------------------------- #
# composer: signal gates + quotas
# --------------------------------------------------------------------------- #
def _probe(sid: str, *, frac_good: float, std: float, gt: str = "x" * 40) -> dict:
    n = 8
    k = int(round(frac_good * n))
    preds = [gt if i < k else "wrong" for i in range(n)]
    return {"sample_id": sid, "target": gt, "is_negative": False, "completions": preds}


def test_score_probe_gates():
    good = composer.score_probe(_probe("a", frac_good=0.5, std=0.2), good_frac=0.8)
    assert composer.is_selectable(good, min_good_frac=0.125, max_good_frac=0.875, min_std=0.03)
    saturated = composer.score_probe(_probe("b", frac_good=1.0, std=0.0), good_frac=0.8)
    assert not composer.is_selectable(saturated, min_good_frac=0.125, max_good_frac=0.875, min_std=0.03)
    collapsed = composer.score_probe(_probe("c", frac_good=0.0, std=0.0), good_frac=0.8)
    assert not composer.is_selectable(collapsed, min_good_frac=0.125, max_good_frac=0.875, min_std=0.03)


def test_score_probe_rejects_negatives():
    rec = _probe("n", frac_good=0.5, std=0.2)
    rec["is_negative"] = True
    sc = composer.score_probe(rec, good_frac=0.8)
    assert not sc["selectable"]


def test_pick_two_per_page_balances_tertiles_and_pages():
    cands = []
    for i in range(30):
        row = _row("multi_frag", "x" * 40, page=f"pg{i % 10}", tmpl=f"t{i % 5}", sid=f"p{i}")
        row["metadata"]["grp_tertile"] = i % 3
        cands.append({"value": float(i), "reward_std": 0.1, "_row": row})
    picked = composer.pick_two_per_page(cands, 12, page_cap=2)
    assert len(picked) == 12
    from collections import Counter

    assert max(Counter(c["_row"]["metadata"]["page_id"] for c in picked).values()) <= 2
    tert = Counter(c["_row"]["metadata"]["grp_tertile"] for c in picked)
    assert max(tert.values()) <= 4 + 1  # ceil(12/3)=4, small tolerance


def test_hq_mix_quotas_sum_to_one():
    assert pytest.approx(sum(composer.HQ_MIX.values()), abs=1e-9) == 1.0
    assert pytest.approx(sum(builder.DEFAULT_MIX.values()), abs=1e-9) == 1.0


# --------------------------------------------------------------------------- #
# Q2 kind: source extraction, reward weights, gate
# --------------------------------------------------------------------------- #
def _q2_row(pool: str, source: str, translation: str = "译文", *, page: str = "p1",
            tmpl: str = "t1", area: float = 0.0014, n_frag: int = 1) -> dict:
    gt = f"<source>\n{source}\n</source>\n<translation>\n{translation}\n</translation>"
    r = _row(pool, gt, page=page, tmpl=tmpl, area=area, n_frag=n_frag)
    return r


def test_q2_gt_text_uses_source_only():
    row = _q2_row("multi_frag", "hello block")
    assert builder.gt_text(row, "q1").startswith("<source>")
    assert builder.gt_text(row, "q2").strip() == "hello block"


def test_q2_reason_uses_source_text():
    long_source = " ".join(f"w{i}" for i in range(120))
    assert builder.reason_of(_q2_row("core_inner", long_source), long_threshold=100, kind="q2") == "long_block"
    assert builder.reason_of(_q2_row("core_inner", "short source"), long_threshold=100, kind="q2") is None
    assert builder.reason_of(_q2_row("real_labeled", "x"), long_threshold=100, kind="q2") == "real"


def test_weights_from_env_and_ceiling(monkeypatch):
    from point_ocr.grpo_rewards_q2 import positive_ceiling, weights_from_env

    monkeypatch.delenv("GRPO_REWARD_WEIGHTS", raising=False)
    assert weights_from_env().translation == 1.0  # default

    monkeypatch.setenv("GRPO_REWARD_WEIGHTS", "1.5,0.05,1.0,1.5,1.5,0.75,0.5")
    w = weights_from_env()
    assert w.source == 1.5 and w.translation == 0.05
    assert abs(positive_ceiling(w) - 2.55) < 1e-9

    monkeypatch.setenv("GRPO_REWARD_WEIGHTS", "1,2,3")
    with pytest.raises(ValueError):
        weights_from_env()


def test_q2_score_probe_value_prefers_localisation_failure(monkeypatch):
    monkeypatch.setenv("GRPO_REWARD_WEIGHTS", "1.5,0.05,1.0,1.5,1.5,0.75,0.5")
    gt = "<source>\nhello world\n</source>\n<translation>\n你好世界\n</translation>"
    over_src = " dumps the whole neighbouring paragraph " * 20  # >=400 chars -> over-extraction
    over = f"<source>\nhello world{over_src}\n</source>\n<translation>\n你好世界\n</translation>"
    perfect = gt
    rec_over = {"sample_id": "a", "target": gt, "is_negative": False,
                "completions": [over] * 4 + [perfect] * 4}
    rec_sat = {"sample_id": "b", "target": gt, "is_negative": False,
               "completions": [perfect] * 8}
    sc_over = composer.score_probe(rec_over, good_frac=0.8, kind="q2")
    sc_sat = composer.score_probe(rec_sat, good_frac=0.8, kind="q2")
    assert sc_over["ceiling"] == pytest.approx(2.55)
    assert composer.is_selectable(sc_over, min_good_frac=0.125, max_good_frac=0.875, min_std=0.03)
    # saturated group has zero variance -> dropped
    assert not composer.is_selectable(sc_sat, min_good_frac=0.125, max_good_frac=0.875, min_std=0.03)
    assert sc_over["value"] > 0.0


# --------------------------------------------------------------------------- #
# "high quality" gate: near-perfect best rollout + real spread
# --------------------------------------------------------------------------- #
def _gate_rec(**kw):
    base = {"selectable": True, "frac_good": 0.5, "reward_std": 0.2, "reward_max_frac": 1.0}
    base.update(kw)
    return base


def test_gate_requires_near_perfect_max_reward():
    ok = dict(min_good_frac=0.125, max_good_frac=0.875, min_std=0.10, min_reward_max_frac=0.95)
    assert composer.is_selectable(_gate_rec(), **ok)
    assert not composer.is_selectable(_gate_rec(reward_max_frac=0.90), **ok)   # best rollout not good enough
    assert not composer.is_selectable(_gate_rec(reward_std=0.05), **ok)        # too little spread
    assert not composer.is_selectable(_gate_rec(frac_good=1.0), **ok)          # saturated
    assert not composer.is_selectable(_gate_rec(frac_good=0.0), **ok)          # all bad


def test_score_probe_exposes_reward_max_frac():
    gt = "hello world block"
    rec = {"sample_id": "x", "target": gt, "is_negative": False,
           "completions": [gt] + ["zzz"] * 7}
    sc = composer.score_probe(rec, good_frac=0.8, kind="q1")
    assert sc["reward_max_frac"] == pytest.approx(1.0)
    assert sc["reward_std"] > 0.10
    assert composer.is_selectable(sc, min_good_frac=0.125, max_good_frac=0.875,
                                 min_std=0.10, min_reward_max_frac=0.95)

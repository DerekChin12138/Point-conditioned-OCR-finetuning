"""Absolute image paths in split JSONLs must be relocatable after a move."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from relocate_paths import relocate_one  # noqa: E402


def test_relocates_when_old_missing_and_new_present(tmp_path: Path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    (new / "data" / "pools_q2" / "core_inner").mkdir(parents=True)
    target = new / "data" / "pools_q2" / "core_inner" / "a.jpg"
    target.write_bytes(b"x")

    raw = str(old / "data" / "pools_q2" / "core_inner" / "a.jpg")  # does not exist
    assert relocate_one(raw, str(old), str(new)) == str(target)


def test_noop_when_old_path_still_exists(tmp_path: Path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    (old / "d").mkdir(parents=True)
    (old / "d" / "a.jpg").write_bytes(b"x")
    raw = str(old / "d" / "a.jpg")
    # old path is valid on this machine -> leave it alone
    assert relocate_one(raw, str(old), str(new)) is None


def test_noop_for_unrelated_or_relative_paths(tmp_path: Path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    new.mkdir(parents=True, exist_ok=True)
    (new / "a.jpg").write_bytes(b"x")
    assert relocate_one(str(tmp_path / "other" / "a.jpg"), str(old), str(new)) is None
    assert relocate_one("data/pools_q2/rel.jpg", str(old), str(new)) is None
    assert relocate_one("", str(old), str(new)) is None


def test_noop_when_target_missing(tmp_path: Path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    raw = str(old / "d" / "gone.jpg")
    assert relocate_one(raw, str(old), str(new)) is None

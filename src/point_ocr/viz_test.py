"""Helpers for notebook visual QA on the test split."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_test_examples(
    test_jsonl: Path,
    *,
    n: int = 5,
    seed: int = 0,
    prefer_positive: bool = True,
) -> list[dict[str, Any]]:
    rows = load_jsonl(test_jsonl)
    rng = random.Random(seed)
    if prefer_positive:
        pos = [r for r in rows if not (r.get("metadata") or {}).get("is_negative")]
        pool = pos if len(pos) >= n else rows
    else:
        pool = rows
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def row_image_and_target(row: dict[str, Any]) -> tuple[str, str, str]:
    img = (row.get("images") or [""])[0]
    msgs = row.get("messages") or []
    target = msgs[1]["content"] if len(msgs) > 1 else ""
    sid = (row.get("metadata") or {}).get("sample_id", "")
    return str(img), str(target), str(sid)

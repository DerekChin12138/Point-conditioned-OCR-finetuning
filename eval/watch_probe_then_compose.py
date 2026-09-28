#!/usr/bin/env python3
"""Wait until the GRPO value probe has N scored rows, then compose the next split.

Does not stop the probe. Does not start training.

  uv run python eval/watch_probe_then_compose.py --min-probe 1000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _n_scored(summary: Path, rows: Path) -> int:
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            n = int(payload.get("n_scored") or 0)
            if n:
                return n
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if not rows.is_file():
        return 0
    n = 0
    with rows.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-probe", type=int, default=1000)
    ap.add_argument("--poll-sec", type=float, default=60)
    ap.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "checkpoints/grpo_value_probe/summary.json",
    )
    ap.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "checkpoints/grpo_value_probe/rows.jsonl",
    )
    args = ap.parse_args()

    print(
        f"watch probe until n_scored>={args.min_probe} (poll {args.poll_sec:.0f}s)",
        flush=True,
    )
    while True:
        n = _n_scored(args.summary, args.rows)
        print(f"probe n_scored={n}/{args.min_probe}", flush=True)
        if n >= args.min_probe:
            break
        time.sleep(max(5.0, float(args.poll_sec)))

    cmd = [
        sys.executable,
        str(ROOT / "data/scripts/compose_grpo_from_probe.py"),
        "--min-probe",
        str(args.min_probe),
    ]
    print("compose:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Zero-shot baseline: stock OvisOCR2 + POINT prompt (expect high over-extraction)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "eval" / "heldout" / "manifest.json")
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eval" / "results" / "zero_shot_ovis.json",
    )
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "eval" / "run_eval.py"),
        "--manifest",
        str(args.manifest),
        "--backend",
        "vllm",
        "--model",
        args.model,
        "--out",
        str(args.out),
        "--pred-out",
        str(args.out.with_suffix(".pred.jsonl")),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

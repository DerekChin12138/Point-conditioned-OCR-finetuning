#!/usr/bin/env python3
"""Compose the first GRPO split from q1_withreal SFT jsonl.

Does not re-render. Image paths stay inside pools_q / pools_real.

  uv run python data/scripts/compose_grpo_q1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.grpo_data import compose_grpo_split  # noqa: E402

DEFAULT_RECIPE = ROOT / "data/recipes/grpo_q1_hard.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise SystemExit(f"recipe {args.recipe} is not a mapping")
    src = args.src or ROOT / str(recipe.get("from") or "data/splits_stage_q1_withreal")
    out = args.out or (ROOT / "data" / f"splits_{recipe.get('name') or 'grpo_q1_hard'}")
    meta = compose_grpo_split(src_dir=src, out_dir=out, recipe=recipe)
    print(json.dumps({k: meta[k] for k in ("name", "n_train", "n_val", "train_by_scene", "val_by_scene", "train")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rewrite absolute image paths inside split JSONLs after moving the workspace.

The `data/splits_*/**.jsonl` rows store absolute image paths under the machine
that built them (e.g. ``/home/<user>/.../data/pools_q2/...``). After uploading to
another host / another root, those paths no longer exist.

This rewrites a path only when ALL of these hold:
  * it starts with ``--old-root``
  * the file at the old path does NOT exist
  * the file at the relocated path DOES exist

so it is a no-op on the machine that produced the data.

  # dry run (default)
  uv run python scripts/relocate_paths.py --old-root /home/me/point-ocr-finetuning

  # apply
  uv run python scripts/relocate_paths.py --old-root /home/me/point-ocr-finetuning --apply
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_GLOBS = (
    "data/splits_*/*.jsonl",
)


def iter_split_files(globs: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for g in globs:
        out.extend(sorted(ROOT.glob(g)))
    return out


def relocate_one(raw: str, old_root: str, new_root: str) -> str | None:
    if not raw or not os.path.isabs(raw) or not raw.startswith(old_root):
        return None
    if os.path.exists(raw):
        return None  # already valid on this machine
    rel = os.path.relpath(raw, old_root)
    cand = os.path.join(new_root, rel)
    if os.path.exists(cand):
        return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-root", required=True, help="Absolute project root baked into the JSONLs.")
    ap.add_argument("--new-root", default=str(ROOT), help="This machine's project root (default: repo root).")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    ap.add_argument("--glob", action="append", default=None, help="Split glob (repeatable).")
    args = ap.parse_args()

    old_root = os.path.abspath(args.old_root)
    new_root = os.path.abspath(args.new_root)
    globs = tuple(args.glob) if args.glob else DEFAULT_SPLIT_GLOBS
    files = iter_split_files(globs)
    if not files:
        raise SystemExit(f"no split files matched {globs} under {ROOT}")

    total_rows = total_changed = 0
    changed_files = 0
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        out_lines: list[str] = []
        n_changed = 0
        for line in lines:
            if not line.strip():
                out_lines.append(line)
                continue
            row = json.loads(line)
            imgs = row.get("images") or []
            if imgs:
                newp = relocate_one(imgs[0], old_root, new_root)
                if newp is not None:
                    row["images"] = [newp] + list(imgs[1:])
                    n_changed += 1
            out_lines.append(json.dumps(row, ensure_ascii=False))
        total_rows += len(lines)
        if n_changed:
            changed_files += 1
            total_changed += n_changed
            if args.apply:
                f.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"  {f.relative_to(ROOT)}: {n_changed}/{len(lines)} rows relocated")
    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\n{verb} {total_changed} rows across {changed_files} files (scanned {total_rows} rows, {len(files)} files)")
    print(f"old_root={old_root}\nnew_root={new_root}")
    if not args.apply and total_changed:
        print("\nre-run with --apply to write.")


if __name__ == "__main__":
    main()

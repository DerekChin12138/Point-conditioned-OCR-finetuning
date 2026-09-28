"""Doc hygiene: repo paths and doc links referenced by the handover docs must exist.

Guards against the failure mode found on 2026-09-28 — README kept pointing at
`checkpoints/q1`, `checkpoints/grpo_q1_hq200/`, `data/splits_grpo_q2_ocr_mt/`
after those were deleted/renamed. Only *code/doc* paths are checked; generated
artifacts under data/, checkpoints/, models/ are excluded (they may not exist yet).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs/HANDOVER_2026-09-28.md"]
CHECK_PREFIXES = ("src/", "train/", "eval/", "export/", "scripts/", "tests/", "docs/")
BACKTICKED = re.compile(r"`([A-Za-z0-9_./-]+)`")
MD_LINK = re.compile(r"\]\(([^)#\s]+\.md)\)")


def _doc_text(path: Path) -> str:
    assert path.is_file(), f"handover doc missing: {path}"
    return path.read_text(encoding="utf-8")


def test_handover_docs_exist():
    for p in DOCS:
        assert p.is_file()


def test_markdown_links_resolve():
    for doc in DOCS:
        for link in MD_LINK.findall(_doc_text(doc)):
            if link.startswith(("http://", "https://")):
                continue
            target = (doc.parent / link).resolve()
            assert target.is_file(), f"{doc.name} → broken link {link}"


def test_referenced_code_paths_exist():
    missing: list[str] = []
    for doc in DOCS:
        for path in set(BACKTICKED.findall(_doc_text(doc))):
            if not path.startswith(CHECK_PREFIXES):
                continue
            if "/checkpoint-" in path:  # e.g. checkpoints/<run>/checkpoint-N
                continue
            if not (ROOT / path).exists():
                missing.append(f"{doc.name}: {path}")
    assert not missing, "docs reference missing code paths:\n  " + "\n  ".join(sorted(missing))

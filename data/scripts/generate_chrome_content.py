#!/usr/bin/env python3
"""Fill a reusable window-chrome copy pool (OpenAI-compatible API).

Chrome is layer 3 of the synth engine: tabs, URLs, filenames, status-bar
snippets — not document body text. Body copy still comes from a1_pool.json.

Env (.env): POINT_OCR_LLM_BASE_URL, POINT_OCR_LLM_API_KEY, POINT_OCR_LLM_MODEL

  uv run python data/scripts/generate_chrome_content.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from generate_template_content import _load_dotenv, chat_json  # noqa: E402

_BROWSER_PROMPT = """\
Generate {n} realistic desktop-browser window scenes for screenshot UI chrome
(NOT the page article). Mix Chinese and English users, mix academic / work /
casual browsing. Each scene is one object:
- tabs: 3-5 short tab titles (as shown on the tab strip)
- active: index of the selected tab (0-based)
- url: full https URL matching the active tab
- bookmarks: 3-6 bookmark bar labels
- profile: short display name
JSON: {{"scenes":[{{"tabs":[...],"active":0,"url":"...","bookmarks":[...],"profile":"..."}}]}}
No markdown fences. Do not mention OCR, datasets, or training.
"""

_OFFICE_PROMPT = """\
Generate {n} realistic Word / WPS 文字 window scenes for screenshot UI chrome
(title bar, ribbon, status bar). Mix zh and en. Each scene:
- app: "WPS 文字" or "Word" or "Microsoft Word"
- filename: plausible .docx name
- user: short person name
- page: like "3 / 18"
- words: integer as string
- zoom: like "100%"
- lang: "zh" or "en"
- outline: 3-6 navigation pane headings
JSON: {{"scenes":[{{"app":"...","filename":"...","user":"...","page":"...","words":"...","zoom":"...","lang":"zh","outline":[...]}}]}}
No markdown fences. Do not mention OCR or training.
"""


def main() -> None:
    _load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/synth/content_pools/chrome_pool.json")
    ap.add_argument("--base-url", default=os.environ.get("POINT_OCR_LLM_BASE_URL", ""))
    ap.add_argument("--api-key", default=os.environ.get("POINT_OCR_LLM_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("POINT_OCR_LLM_MODEL", ""))
    ap.add_argument("--n-browser", type=int, default=24)
    ap.add_argument("--n-office", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()
    if not args.base_url or not args.api_key or not args.model:
        raise SystemExit("Set POINT_OCR_LLM_BASE_URL / API_KEY / MODEL in .env")

    browser = chat_json(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        user=_BROWSER_PROMPT.format(n=args.n_browser),
        temperature=args.temperature,
        timeout_s=args.timeout,
    )
    office = chat_json(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        user=_OFFICE_PROMPT.format(n=args.n_office),
        temperature=args.temperature,
        timeout_s=args.timeout,
    )
    payload = {
        "model": args.model,
        "browser_scenes": browser.get("scenes") or browser.get("items") or [],
        "office_scenes": office.get("scenes") or office.get("items") or [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {args.out} browser={len(payload['browser_scenes'])} office={len(payload['office_scenes'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()

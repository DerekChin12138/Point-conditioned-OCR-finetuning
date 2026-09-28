#!/usr/bin/env python3
"""Generate / grow a reusable text pool (OpenAI-compatible API).

Calls the model in small batches and merges unique strings into --out
(so 2000 paragraphs is many short calls, not one huge JSON).

Env (.env): POINT_OCR_LLM_BASE_URL, POINT_OCR_LLM_API_KEY, POINT_OCR_LLM_MODEL
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def chat_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user: str,
    temperature: float,
    timeout_s: float,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "Return compact JSON only. No markdown fences."},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:600]}") from e
    raw = payload["choices"][0]["message"]["content"]
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.M).strip()
    return json.loads(text)


_THEMES = [
    "urban commuting and transit delays",
    "kitchen experiments and home cooking",
    "software product changelog and docs",
    "field biology and weather notes",
    "museum exhibits and local history",
    "clinic visits and public health tips",
    "small-business retail operations",
    "university lab and course logistics",
    "travel delays and lodging mishaps",
    "sports practice and weekend leagues",
    "gardening, soil, and balcony plants",
    "music rehearsal and venue logistics",
    "library archives and citation notes",
    "harbor shipping and warehouse shifts",
    "film production call sheets",
    "municipal permits and neighborhood news",
    "coffee shop operations and suppliers",
    "hiking trail reports and gear notes",
    "language-learning classroom scenes",
    "apartment maintenance and landlord notices",
]

_BUCKET_PROMPTS = {
    "headings": (
        "Generate {n} short document headings (3-12 words). "
        "Topics: tech/product/news/docs/life. Mix zh and en. Diverse, no repeats. "
        "Theme cluster: {theme}. "
        'JSON: {{"items":["..."]}}'
    ),
    "paragraphs": (
        "Generate {n} short paragraphs (40-90 English words OR 60-120 Chinese chars). "
        "Realistic scenes. Mix zh/en. No OCR/training talk. No duplicate plots. "
        "Theme cluster: {theme}. "
        'JSON: {{"items":["..."]}}'
    ),
    "list_items": (
        "Generate {n} short list/bullet lines (one sentence). Mix zh/en. Distinct. "
        "Theme cluster: {theme}. "
        'JSON: {{"items":["..."]}}'
    ),
    "captions": (
        "Generate {n} short UI captions/meta lines (under 20 words). Mix zh/en. "
        "Theme cluster: {theme}. "
        'JSON: {{"items":["..."]}}'
    ),
    "code_lines": (
        "Generate {n} single-line code-like snippets (python/shell/yaml-ish). Distinct. "
        "Theme cluster: {theme}. "
        'JSON: {{"items":["..."]}}'
    ),
    "cells": (
        "Generate {n} tiny table cell values (numbers, short labels, dates). Distinct. "
        "Theme cluster: {theme}. "
        'JSON: {{"items":["..."]}}'
    ),
}


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    seen = {s.strip() for s in existing}
    out = list(existing)
    for x in incoming:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _save(path: Path, model: str, pool: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "pool": pool,
        "counts": {k: len(v) for k, v in pool.items()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    _load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/synth/content_pools/a1_pool.json")
    ap.add_argument("--base-url", default=os.environ.get("POINT_OCR_LLM_BASE_URL", ""))
    ap.add_argument("--api-key", default=os.environ.get("POINT_OCR_LLM_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("POINT_OCR_LLM_MODEL", ""))
    ap.add_argument("--n-headings", type=int, default=24)
    ap.add_argument("--n-paragraphs", type=int, default=40)
    ap.add_argument("--n-list-items", type=int, default=30)
    ap.add_argument("--n-captions", type=int, default=20)
    ap.add_argument("--n-code-lines", type=int, default=16)
    ap.add_argument("--n-cells", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=40, help="Items requested per API call")
    ap.add_argument("--workers", type=int, default=4, help="Parallel API calls per bucket")
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--append", action="store_true", help="Grow existing --out instead of replacing")
    args = ap.parse_args()

    if not args.base_url or not args.api_key or not args.model:
        raise SystemExit("Set POINT_OCR_LLM_BASE_URL / API_KEY / MODEL in .env")

    pool: dict[str, list[str]] = {
        "headings": [],
        "paragraphs": [],
        "list_items": [],
        "captions": [],
        "code_lines": [],
        "cells": [],
    }
    if args.append and args.out.is_file():
        old = json.loads(args.out.read_text(encoding="utf-8"))
        old_pool = old.get("pool") or {}
        for k in pool:
            pool[k] = [str(x).strip() for x in (old_pool.get(k) or []) if str(x).strip()]
        print(f"[load] {args.out} counts={ {k: len(v) for k, v in pool.items()} }", flush=True)

    targets = {
        "headings": args.n_headings,
        "paragraphs": args.n_paragraphs,
        "list_items": args.n_list_items,
        "captions": args.n_captions,
        "code_lines": args.n_code_lines,
        "cells": args.n_cells,
    }

    batch_i = 0

    def _one_batch(key: str, ask: int, theme: str) -> list[str]:
        obj = chat_json(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            user=_BUCKET_PROMPTS[key].format(n=ask, theme=theme),
            temperature=args.temperature,
            timeout_s=args.timeout,
        )
        items = obj.get("items") or obj.get(key) or []
        if not isinstance(items, list):
            raise RuntimeError(f"bad payload keys={list(obj)[:8]}")
        return [str(x).strip() for x in items if str(x).strip()]

    for key, target in targets.items():
        while len(pool[key]) < target:
            remaining = target - len(pool[key])
            n_jobs = min(max(1, args.workers), max(1, (remaining + args.batch_size - 1) // args.batch_size))
            jobs: list[tuple[int, str]] = []
            left = remaining
            for _ in range(n_jobs):
                need = min(args.batch_size, left)
                ask = min(args.batch_size, need + 8)
                theme = _THEMES[batch_i % len(_THEMES)]
                jobs.append((ask, theme))
                batch_i += 1
                left -= need
                if left <= 0:
                    break
            print(
                f"[gen] {key} have={len(pool[key])}/{target} jobs={len(jobs)} "
                f"ask={[j[0] for j in jobs]} …",
                flush=True,
            )
            got: list[list[str]] = []
            with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                futs = [ex.submit(_one_batch, key, ask, theme) for ask, theme in jobs]
                for fut in as_completed(futs):
                    try:
                        got.append(fut.result())
                    except Exception as e:
                        print(f"  [warn] {e}; will retry leftover", flush=True)
            if not got:
                time.sleep(2.0)
                continue
            before = len(pool[key])
            for items in got:
                pool[key] = _merge_unique(pool[key], items)
            print(f"  +{len(pool[key]) - before} → {len(pool[key])}", flush=True)
            _save(args.out, args.model, pool)
            if args.sleep > 0:
                time.sleep(args.sleep)
        pool[key] = pool[key][:target]
        _save(args.out, args.model, pool)

    print(f"wrote {args.out} counts={ {k: len(v) for k, v in pool.items()} }")


if __name__ == "__main__":
    main()

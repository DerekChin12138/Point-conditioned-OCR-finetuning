#!/usr/bin/env python3
"""Download OPUS en-zh pairs, length-bucket them, write a bilingual content pool.

Daily dialogue/news/talks are the majority; specialized medical/legal/UN/software
are mixed in. Pairs are unique by normalized English.

  uv run python data/scripts/build_opus_bilingual_pool.py
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "data/synth/content_pools/q2_opus_en_zh.json"

# HuggingFace (opus-100 is the one corpus that actually hosts en-zh there).
HF_SOURCES: list[tuple[str, str, str, int]] = [
    ("Helsinki-NLP/opus-100", "en-zh", "daily_mixed", 140_000),
]

# Official OPUS moses bitexts: https://opus.nlpl.eu/
# (name, url, domain, max_keep)
MOSES_SOURCES: list[tuple[str, str, str, int]] = [
    (
        "TED2020",
        "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-zh_cn.txt.zip",
        "daily_talks",
        35_000,
    ),
    (
        "NeuLab-TedTalks",
        "https://object.pouta.csc.fi/OPUS-NeuLab-TedTalks/v1/moses/en-zh_cn.txt.zip",
        "daily_talks",
        20_000,
    ),
    (
        "MultiUN",
        "https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/en-zh.txt.zip",
        "spec_diplomacy",
        22_000,
    ),
    (
        "GNOME",
        "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/en-zh_CN.txt.zip",
        "spec_software",
        12_000,
    ),
    (
        "Ubuntu",
        "https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/en-zh_CN.txt.zip",
        "spec_software",
        10_000,
    ),
    (
        "KDE4",
        "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-zh_CN.txt.zip",
        "spec_software",
        8_000,
    ),
    (
        "OpenSubtitles",
        "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-zh_cn.txt.zip",
        "daily_dialogue",
        90_000,
    ),
]
CACHE_DIR = ROOT / "data/synth/content_pools/.opus_cache"

BUCKETS = ("headings", "paragraphs", "list_items", "captions", "code_lines", "cells")
_HTML_RE = re.compile(r"<[^>]+>")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _norm_key(en: str) -> str:
    return " ".join(en.casefold().split())


def _looks_en(text: str) -> bool:
    latin = len(_LATIN_RE.findall(text))
    cjk = len(_CJK_RE.findall(text))
    if latin < 2:
        return False
    return latin >= cjk * 2


def _looks_zh(text: str) -> bool:
    return len(_CJK_RE.findall(text)) >= 2


def _clean(text: str) -> str:
    s = (text or "").replace("\u00a0", " ").replace("\r", " ")
    s = _HTML_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _bucket_for(en: str, domain: str) -> str | None:
    n = len(en)
    if n < 2:
        return None
    if n <= 14:
        return "cells"
    if domain.startswith("spec_software") and 12 <= n <= 90:
        return "code_lines"
    if n <= 42:
        return "headings"
    if n <= 70:
        return "captions" if n <= 52 else "list_items"
    if n <= 110:
        return "list_items"
    if n <= 520:
        return "paragraphs"
    return None


def _ok_pair(en: str, zh: str) -> bool:
    if not en or not zh:
        return False
    if len(en) < 2 or len(zh) < 2:
        return False
    if len(en) > 520 or len(zh) > 700:
        return False
    if not _looks_en(en) or not _looks_zh(zh):
        return False
    ratio = len(zh) / max(1, len(en))
    if ratio < 0.28 or ratio > 2.6:
        return False
    if en.casefold() == zh.casefold():
        return False
    lowered = en.casefold()
    if "http://" in lowered or "https://" in lowered:
        return False
    if en.count("{") + en.count("}") > 4:
        return False
    return True


def _iter_rows(ds) -> Iterable[dict[str, Any]]:
    for row in ds:
        yield row


def _extract_en_zh(row: dict[str, Any]) -> tuple[str, str] | None:
    tr = row.get("translation")
    if isinstance(tr, dict):
        en = tr.get("en") or tr.get("eng")
        zh = tr.get("zh") or tr.get("zho") or tr.get("zh-CN") or tr.get("zh_cn")
        if en and zh:
            return str(en), str(zh)
    src = row.get("source") or row.get("src") or row.get("en")
    tgt = row.get("target") or row.get("tgt") or row.get("zh")
    if src and tgt:
        return str(src), str(tgt)
    return None


def _try_load(name: str, config: str, max_keep: int):
    from datasets import load_dataset

    configs = [config, "enzh", "en-zh", "zh-en"]
    last_err: Exception | None = None
    for cfg in configs:
        for split in ("train", "train[:80%]", None):
            try:
                kwargs: dict[str, Any] = {"path": name, "streaming": True}
                if cfg:
                    kwargs["name"] = cfg
                if split:
                    kwargs["split"] = split
                print(f"  load {name} name={cfg} split={split}", flush=True)
                ds = load_dataset(**kwargs)
                if hasattr(ds, "keys") and not split:
                    # DatasetDict
                    if "train" in ds:
                        ds = ds["train"]
                    else:
                        ds = ds[next(iter(ds))]
                return ds
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        # non-streaming fallback
        try:
            print(f"  load-nostream {name} name={cfg}", flush=True)
            ds = load_dataset(name, cfg, split="train")
            return ds
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"failed {name}: {last_err}")


def harvest_source(
    name: str,
    config: str,
    domain: str,
    max_keep: int,
    seen: set[str],
) -> list[dict[str, str]]:
    t0 = time.time()
    kept: list[dict[str, str]] = []
    scanned = 0
    try:
        ds = _try_load(name, config, max_keep)
    except Exception as e:  # noqa: BLE001
        print(f"[skip] {name}: {type(e).__name__}: {e}", flush=True)
        return kept
    try:
        for row in _iter_rows(ds):
            scanned += 1
            pair = _extract_en_zh(row)
            if pair is None:
                # maybe swapped
                continue
            en, zh = _clean(pair[0]), _clean(pair[1])
            if not _looks_en(en) and _looks_en(zh) and _looks_zh(en):
                en, zh = zh, en
            if not _ok_pair(en, zh):
                continue
            key = _norm_key(en)
            if not key or key in seen:
                continue
            bucket = _bucket_for(en, domain)
            if bucket is None:
                continue
            seen.add(key)
            kept.append(
                {
                    "en": en,
                    "zh": zh,
                    "domain": domain,
                    "source": name.split("/")[-1],
                    "bucket": bucket,
                }
            )
            if len(kept) >= max_keep:
                break
            if scanned % 20000 == 0:
                print(
                    f"  … {name} scanned={scanned} kept={len(kept)} elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {name} stopped early: {type(e).__name__}: {e}", flush=True)
    print(
        f"[ok] {name} domain={domain} scanned={scanned} kept={len(kept)} {time.time()-t0:.1f}s",
        flush=True,
    )
    return kept


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1024:
        print(f"  cache hit {dest.name} ({dest.stat().st_size} bytes)", flush=True)
        return dest
    print(f"  GET {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "point-ocr-opus/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        t0 = time.time()
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if time.time() - t0 > 2:
                    if total:
                        print(f"    … {got/1e6:.1f}/{total/1e6:.1f} MB", flush=True)
                    else:
                        print(f"    … {got/1e6:.1f} MB", flush=True)
                    t0 = time.time()
    tmp.replace(dest)
    print(f"  saved {dest.name} ({dest.stat().st_size} bytes)", flush=True)
    return dest


def _pick_moses_pair(names: list[str]) -> tuple[str, str] | None:
    en = [n for n in names if n.endswith(".en") or n.endswith(".eng")]
    zh = [
        n
        for n in names
        if n.endswith(".zh")
        or n.endswith(".zho")
        or n.endswith(".zh_CN")
        or n.endswith(".zh_cn")
    ]
    if en and zh:
        return en[0], zh[0]
    return None


def harvest_moses(
    name: str,
    url: str,
    domain: str,
    max_keep: int,
    seen: set[str],
) -> list[dict[str, str]]:
    t0 = time.time()
    kept: list[dict[str, str]] = []
    scanned = 0
    zip_en = CACHE_DIR / f"{name}.en-zh.txt.zip"
    zip_zh = CACHE_DIR / f"{name}.zh-en.txt.zip"
    alt_urls = [(url, zip_en)]
    if "/en-zh.txt.zip" in url:
        alt_urls.append((url.replace("/en-zh.txt.zip", "/zh-en.txt.zip"), zip_zh))
    last_err: Exception | None = None
    zpath = None
    for u, dest in alt_urls:
        try:
            zpath = _download(u, dest)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  miss {u}: {type(e).__name__}: {e}", flush=True)
    if zpath is None:
        print(f"[skip] {name}: download failed ({last_err})", flush=True)
        return kept
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            pair_names = _pick_moses_pair(names)
            if pair_names is None:
                print(f"[skip] {name}: no en/zh files in {names[:8]}", flush=True)
                return kept
            en_name, zh_name = pair_names
            print(f"  bitext {en_name} || {zh_name}", flush=True)
            with zf.open(en_name) as fe, zf.open(zh_name) as fz:
                en_f = io.TextIOWrapper(fe, encoding="utf-8", errors="replace")
                zh_f = io.TextIOWrapper(fz, encoding="utf-8", errors="replace")
                for en_line, zh_line in zip(en_f, zh_f):
                    scanned += 1
                    en, zh = _clean(en_line), _clean(zh_line)
                    if not _looks_en(en) and _looks_en(zh) and _looks_zh(en):
                        en, zh = zh, en
                    if not _ok_pair(en, zh):
                        continue
                    key = _norm_key(en)
                    if not key or key in seen:
                        continue
                    bucket = _bucket_for(en, domain)
                    if bucket is None:
                        continue
                    seen.add(key)
                    kept.append(
                        {
                            "en": en,
                            "zh": zh,
                            "domain": domain,
                            "source": name,
                            "bucket": bucket,
                        }
                    )
                    if len(kept) >= max_keep:
                        break
                    if scanned % 50000 == 0:
                        print(
                            f"  … {name} scanned={scanned} kept={len(kept)} "
                            f"elapsed={time.time()-t0:.0f}s",
                            flush=True,
                        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {name} stopped early: {type(e).__name__}: {e}", flush=True)
    print(
        f"[ok] {name} domain={domain} scanned={scanned} kept={len(kept)} {time.time()-t0:.1f}s",
        flush=True,
    )
    return kept


def to_pool(pairs: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    bags: dict[str, list[dict[str, str]]] = {b: [] for b in BUCKETS}
    for p in pairs:
        b = p.get("bucket") or _bucket_for(p["en"], p.get("domain") or "")
        if b not in bags:
            continue
        bags[b].append(
            {
                "en": p["en"],
                "zh": p["zh"],
                "domain": p["domain"],
                "source": p["source"],
            }
        )
    return bags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--daily-frac",
        type=float,
        default=0.75,
        help="Target share of daily-domain pairs after harvest (subsample specialized if needed).",
    )
    ap.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip a moses corpus by name (repeatable), e.g. OpenSubtitles.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    all_pairs: list[dict[str, str]] = []
    per_source: dict[str, int] = {}
    for name, config, domain, cap in HF_SOURCES:
        print(f"[hf] {name} {domain} cap={cap}", flush=True)
        got = harvest_source(name, config, domain, cap, seen)
        per_source[name] = len(got)
        all_pairs.extend(got)
    for name, url, domain, cap in MOSES_SOURCES:
        if name in set(args.skip or []):
            print(f"[moses] skip {name}", flush=True)
            continue
        print(f"[moses] {name} {domain} cap={cap}", flush=True)
        got = harvest_moses(name, url, domain, cap, seen)
        per_source[name] = len(got)
        all_pairs.extend(got)

    daily = [p for p in all_pairs if str(p["domain"]).startswith("daily")]
    spec = [p for p in all_pairs if str(p["domain"]).startswith("spec")]
    rng.shuffle(daily)
    rng.shuffle(spec)
    # Prefer keeping all daily; cap specialized so daily stays majority.
    if daily:
        max_spec = int(round(len(daily) * (1.0 - args.daily_frac) / max(args.daily_frac, 1e-6)))
        if len(spec) > max_spec:
            spec = spec[: max(max_spec, 8_000)]
    mixed = daily + spec
    rng.shuffle(mixed)
    pool = to_pool(mixed)
    domain_counts = Counter(p["domain"] for p in mixed)
    bucket_counts = {k: len(v) for k, v in pool.items()}
    meta = {
        "n_pairs": sum(bucket_counts.values()),
        "n_daily": len(daily),
        "n_spec": len(spec),
        "domains": dict(domain_counts),
        "buckets": bucket_counts,
        "sources": per_source,
        "daily_frac_target": args.daily_frac,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "pool": pool}
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {args.out} n={meta['n_pairs']}", flush=True)
    if meta["n_pairs"] < 40_000:
        print("[warn] pool is thinner than expected; check skipped OPUS sources", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build Marker studio galleries: Q1 test pages (unmarked) + small OOD probes + signal multi_frag.

  uv run python eval/marker_studio/build_galleries.py
  uv run python eval/marker_studio/build_galleries.py --skip-ood
  uv run python eval/marker_studio/build_galleries.py --only-signal-mf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TEST_JSONL = ROOT / "data/splits_stage_q1/test.jsonl"
POOLS_ROOT = ROOT / "data/pools_q"
GALLERY = ROOT / "data/studio_galleries"
Q1_DIR = GALLERY / "q1_test"
OOD_DIR = GALLERY / "ood"
SIGNAL_MF_DIR = GALLERY / "signal_multi_frag"
SIGNAL_VAL_JSONL = ROOT / "data/splits_grpo_q1_signal/val.jsonl"

IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_q1_test_pages(
    rows: list[dict[str, Any]], *, max_pages: int = 40
) -> list[tuple[str, dict[str, Any]]]:
    """Unique test pages, one per template, then chrome/pool coverage."""
    by_page: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = row.get("metadata") or {}
        pid = meta.get("page_id")
        if not pid or pid in by_page:
            continue
        by_page[str(pid)] = meta

    by_tmpl: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for pid, meta in by_page.items():
        by_tmpl[str(meta.get("template_stem") or "unknown")].append((pid, meta))

    def _chrome(meta: dict[str, Any]) -> bool:
        fam = meta.get("chrome_family")
        return fam not in (None, "", "none")

    picked: list[tuple[str, dict[str, Any]]] = []
    for _tmpl, items in sorted(by_tmpl.items()):
        items = sorted(
            items,
            key=lambda t: (0 if t[1].get("pool_id") == "core_inner" else 1, t[0]),
        )
        picked.append(items[0])

    have = {pid for pid, _ in picked}
    if len(picked) < max_pages:
        for _tmpl, items in sorted(by_tmpl.items()):
            have_chrome = {
                _chrome(m) for pid, m in picked if m.get("template_stem") == _tmpl
            }
            for pid, meta in items:
                if pid in have:
                    continue
                if _chrome(meta) in have_chrome:
                    continue
                picked.append((pid, meta))
                have.add(pid)
                have_chrome.add(_chrome(meta))
                if len(picked) >= max_pages:
                    break
            if len(picked) >= max_pages:
                break

    pools_have = {m.get("pool_id") for _, m in picked}
    for pid, meta in by_page.items():
        if len(picked) >= max_pages:
            break
        pool = meta.get("pool_id")
        if pool not in pools_have:
            picked.append((pid, meta))
            pools_have.add(pool)

    return picked[:max_pages]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def build_q1_test(*, max_pages: int) -> dict[str, Any]:
    if not TEST_JSONL.is_file():
        raise SystemExit(f"missing {TEST_JSONL}")
    rows = _load_jsonl(TEST_JSONL)
    picked = select_q1_test_pages(rows, max_pages=max_pages)
    Q1_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    missing = 0
    for pid, meta in picked:
        pool = str(meta.get("pool_id") or "")
        src = POOLS_ROOT / pool / "renders" / f"{pid}.png"
        if not src.is_file():
            missing += 1
            print(f"[q1_test] missing render {src}", flush=True)
            continue
        name = f"{pid}.png"
        dst = Q1_DIR / name
        _link_or_copy(src, dst)
        manifest.append(
            {
                "file": name,
                "page_id": pid,
                "pool_id": pool,
                "template_stem": meta.get("template_stem"),
                "chrome_family": meta.get("chrome_family"),
                "source": str(src),
            }
        )
    (Q1_DIR / "manifest.json").write_text(
        json.dumps({"n": len(manifest), "pages": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[q1_test] {len(manifest)} pages → {Q1_DIR} (missing {missing})", flush=True)
    return {"n": len(manifest), "missing": missing}


def _downscale(img: Image.Image, max_side: int = 2880) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    s = max_side / m
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)


def _save_probe(img: Image.Image, path: Path, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _downscale(img).save(path, format="PNG")
    path.with_suffix(".json").write_text(
        json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _take_streaming(ds, n: int):
    i = 0
    for row in ds:
        yield row
        i += 1
        if i >= n:
            return


def pull_screenparse(n: int) -> int:
    from datasets import load_dataset

    out = OOD_DIR / "screenparse"
    print(f"[ood] ScreenParse n={n}", flush=True)
    ds = load_dataset("docling-project/screenparse", split="train", streaming=True)
    saved = 0
    for row in _take_streaming(ds, max(n * 20, 500)):
        img = row.get("image")
        if img is None:
            continue
        w = int(row.get("width") or getattr(img, "size", (0, 0))[0] or 0)
        h = int(row.get("height") or getattr(img, "size", (0, 0))[1] or 0)
        if min(w, h) < 400:
            continue
        texts = row.get("texts") or []
        if not any(str(t).strip() for t in texts[:8]):
            continue
        name = f"{saved:02d}.png"
        _save_probe(
            img,
            out / name,
            {
                "source": "docling-project/screenparse",
                "id": row.get("id"),
                "url": row.get("url"),
                "size": [w, h],
                "n_elements": row.get("num_elements"),
                "sample_texts": [str(t)[:80] for t in texts[:6] if str(t).strip()],
            },
        )
        saved += 1
        print(f"  screenparse {saved}/{n} ({w}x{h})", flush=True)
        if saved >= n:
            break
    return saved


def pull_webui(n: int) -> int:
    from datasets import load_dataset

    out = OOD_DIR / "webui"
    print(f"[ood] WebUI n={n}", flush=True)
    ds = load_dataset("ronantakizawa/webui", split="train", streaming=True)
    saved = 0
    for row in _take_streaming(ds, max(n * 20, 500)):
        viewport = str(row.get("viewport") or "")
        if viewport and viewport not in {"desktop", "1280x720"}:
            # keep a mix: prefer desktop, allow others if we are short
            if saved < n // 2:
                continue
        img = row.get("image")
        if img is None:
            continue
        w, h = img.size
        if min(w, h) < 300:
            continue
        name = f"{saved:02d}_{viewport or 'unk'}.png"
        _save_probe(
            img,
            out / name,
            {
                "source": "ronantakizawa/webui",
                "sample_id": row.get("sample_id"),
                "viewport": viewport,
                "source_url": row.get("source_url"),
                "component_type": row.get("component_type"),
                "size": [w, h],
            },
        )
        saved += 1
        print(f"  webui {saved}/{n} {viewport} ({w}x{h})", flush=True)
        if saved >= n:
            break
    return saved


def pull_doclaynet(n: int) -> int:
    from datasets import load_dataset

    out = OOD_DIR / "doclaynet"
    print(f"[ood] DocLayNet n={n}", flush=True)
    ds = load_dataset("docling-project/DocLayNet-v1.1", split="train", streaming=True)
    saved = 0
    for row in _take_streaming(ds, max(n * 20, 500)):
        img = row.get("image")
        if img is None:
            continue
        cells = row.get("pdf_cells") or []
        n_cells = sum(len(c) for c in cells) if cells and isinstance(cells[0], list) else len(cells)
        if n_cells == 0:
            continue
        w, h = img.size
        cat = row.get("category_id") or row.get("metadata") or {}
        name = f"{saved:02d}.png"
        _save_probe(
            img,
            out / name,
            {
                "source": "docling-project/DocLayNet-v1.1",
                "size": [w, h],
                "n_pdf_cells": n_cells,
                "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                "category_id": cat if not isinstance(cat, dict) else None,
            },
        )
        saved += 1
        print(f"  doclaynet {saved}/{n} ({w}x{h}) cells={n_cells}", flush=True)
        if saved >= n:
            break
    return saved


def build_ood(*, per_source: int) -> dict[str, int]:
    OOD_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    pullers = (
        ("screenparse", pull_screenparse),
        ("webui", pull_webui),
        ("doclaynet", pull_doclaynet),
    )
    for name, fn in pullers:
        try:
            counts[name] = fn(per_source)
        except Exception as e:  # noqa: BLE001
            counts[name] = 0
            print(f"[ood] {name} failed: {type(e).__name__}: {e}", flush=True)
    (OOD_DIR / "manifest.json").write_text(
        json.dumps({"per_source": per_source, "counts": counts}, indent=2),
        encoding="utf-8",
    )
    print(f"[ood] done {counts}", flush=True)
    return counts


def _assistant_target(row: dict[str, Any]) -> str:
    for msg in row.get("messages") or []:
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def build_signal_multi_frag(*, jsonl: Path | None = None) -> dict[str, Any]:
    """Unmarked pages for signal-val multi_frag (studio places its own X).

    Falls back to the ShareGPT marked jpg when the pool render is missing.
    Sidecar JSON keeps the training click + GT text for reference.
    """
    src = jsonl or SIGNAL_VAL_JSONL
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    rows = _load_jsonl(src)
    SIGNAL_MF_DIR.mkdir(parents=True, exist_ok=True)
    # Clear previous gallery images / sidecars (keep directory).
    for old in SIGNAL_MF_DIR.iterdir():
        if old.suffix.lower() in IMG_SUFFIXES | {".json"} and old.name != "manifest.json":
            old.unlink()

    manifest: list[dict[str, Any]] = []
    missing = 0
    idx = 0
    for row in rows:
        meta = row.get("metadata") or {}
        scene = str(meta.get("grpo_scene") or meta.get("pool_id") or "")
        if scene != "multi_frag":
            continue
        idx += 1
        page_id = str(meta.get("page_id") or f"row_{idx}")
        pool_id = str(meta.get("pool_id") or "multi_frag")
        sample_id = str(meta.get("sample_id") or page_id)
        render = ROOT / "data" / "pools_q" / pool_id / "renders" / f"{page_id}.png"
        images = row.get("images") or []
        marked = Path(str(images[0])) if images else Path()
        if render.is_file():
            src_img, kind = render, "unmarked_render"
            ext = ".png"
        elif marked.is_file():
            src_img, kind = marked, "marked_fallback"
            ext = marked.suffix.lower() if marked.suffix else ".jpg"
        else:
            missing += 1
            print(f"[signal_multi_frag] missing {page_id}", flush=True)
            continue
        safe = sample_id.replace(":", "__").replace("/", "_")
        name = f"{idx:02d}__{safe}{ext}"
        dst = SIGNAL_MF_DIR / name
        _link_or_copy(src_img, dst)
        point = meta.get("point")
        entry = {
            "file": name,
            "sample_id": sample_id,
            "page_id": page_id,
            "pool_id": pool_id,
            "block_id": meta.get("block_id"),
            "point": point,
            "image_w": meta.get("image_w"),
            "image_h": meta.get("image_h"),
            "n_fragments": meta.get("n_fragments"),
            "source_kind": kind,
            "source": str(src_img),
            "target": _assistant_target(row),
        }
        (SIGNAL_MF_DIR / f"{Path(name).stem}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest.append(entry)
        print(f"[signal_multi_frag] {name} ({kind})", flush=True)

    (SIGNAL_MF_DIR / "manifest.json").write_text(
        json.dumps(
            {"n": len(manifest), "source_jsonl": str(src), "pages": manifest},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[signal_multi_frag] {len(manifest)} → {SIGNAL_MF_DIR} (missing {missing})",
        flush=True,
    )
    return {"n": len(manifest), "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-q1-pages", type=int, default=40)
    ap.add_argument("--ood-per-source", type=int, default=8)
    ap.add_argument("--skip-ood", action="store_true")
    ap.add_argument("--skip-q1", action="store_true")
    ap.add_argument("--skip-signal-mf", action="store_true")
    ap.add_argument("--only-signal-mf", action="store_true")
    args = ap.parse_args()
    GALLERY.mkdir(parents=True, exist_ok=True)
    if args.only_signal_mf:
        build_signal_multi_frag()
        return
    if not args.skip_q1:
        build_q1_test(max_pages=args.max_q1_pages)
    if not args.skip_ood:
        build_ood(per_source=args.ood_per_source)
    if not args.skip_signal_mf:
        build_signal_multi_frag()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()

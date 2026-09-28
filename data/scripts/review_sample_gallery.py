#!/usr/bin/env python3
"""Build a local HTML gallery to human-review POINT samples.

Shows image and ground-truth text side-by-side so you can check the label
against the crosshair without scrolling away from the screenshot.

Example:
  uv run python data/scripts/review_sample_gallery.py \\
    --jsonl data/processed/preview_a1_dense/point_sharegpt.jsonl \\
    --n 40 --out data/review_galleries/preview_a1_dense
"""

from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.marker import MARKER_SPEC  # noqa: E402


def _area_ratio(bbox: list | None, w: int, h: int) -> float | None:
    if not bbox or len(bbox) < 4 or w <= 0 or h <= 0:
        return None
    x0, y0, x1, y1 = map(float, bbox[:4])
    return max(0.0, (x1 - x0) * (y1 - y0)) / float(w * h)


def _overlay(
    src: Path,
    dst: Path,
    point: list | None,
    bbox: list | None,
    *,
    show_bbox: bool,
    show_point: bool,
    bboxes: list | None = None,
) -> None:
    """Copy/annotate for review only — never used as training pixels."""
    im = Image.open(src).convert("RGBA")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not show_bbox and not show_point:
        im.convert("RGB").save(dst, quality=92)
        return
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    boxes: list[list] = []
    if show_bbox:
        if isinstance(bboxes, list) and bboxes:
            boxes = [b for b in bboxes if isinstance(b, list) and len(b) >= 4]
        elif bbox and len(bbox) >= 4:
            boxes = [bbox]
        for bi, box in enumerate(boxes):
            x0, y0, x1, y1 = map(int, box[:4])
            # First fragment green; extras teal so multi-rect is obvious
            color = (0, 200, 80, 220) if bi == 0 else (0, 180, 200, 220)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
    if show_point and point and len(point) >= 2:
        cx, cy = int(round(point[0])), int(round(point[1]))
        r = max(8, MARKER_SPEC.ring_radius_px // 2)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 200, 0, 230), width=2)
        draw.line([(cx - 12, cy), (cx + 12, cy)], fill=(255, 200, 0, 230), width=2)
        draw.line([(cx, cy - 12), (cx, cy + 12)], fill=(255, 200, 0, 230), width=2)
    out = Image.alpha_composite(im, overlay).convert("RGB")
    out.save(dst, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--positives-only", action="store_true")
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument(
        "--show-bbox",
        action="store_true",
        help="Overlay green label bbox (debug only; NOT in training images)",
    )
    ap.add_argument(
        "--show-point",
        action="store_true",
        help="Overlay yellow debug cross (training image already has magenta marker)",
    )
    ap.add_argument("--copy-original", action="store_true")
    ap.add_argument(
        "--stratify",
        type=str,
        default="",
        help="Metadata key to stratify gallery picks (e.g. a2_slice)",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    candidates: list[dict] = []
    for r in rows:
        msgs = r.get("messages") or []
        if len(msgs) < 2:
            continue
        gt = msgs[1].get("content", "")
        if not isinstance(gt, str):
            gt = str(gt)
        is_neg = bool((r.get("metadata") or {}).get("is_negative")) or not gt.strip()
        if args.positives_only and is_neg:
            continue
        if args.min_chars and (is_neg or len(gt.strip()) < args.min_chars):
            continue
        imgs = r.get("images") or []
        if not imgs or not Path(imgs[0]).is_file():
            continue
        candidates.append(r)

    if not candidates:
        raise SystemExit("No eligible samples found")

    rng = random.Random(args.seed)
    n = min(args.n, len(candidates))
    if args.stratify:
        buckets: dict[str, list[dict]] = {}
        for r in candidates:
            key = str((r.get("metadata") or {}).get(args.stratify) or "other")
            buckets.setdefault(key, []).append(r)
        picked: list[dict] = []
        keys = sorted(buckets.keys())
        # Round-robin so every slice appears
        while len(picked) < n and keys:
            progressed = False
            for k in list(keys):
                if not buckets[k]:
                    keys.remove(k)
                    continue
                picked.append(buckets[k].pop(rng.randrange(len(buckets[k]))))
                progressed = True
                if len(picked) >= n:
                    break
            if not progressed:
                break
    else:
        picked = rng.sample(candidates, n)

    img_dir = args.out / "images"
    if args.out.exists():
        shutil.rmtree(args.out)
    img_dir.mkdir(parents=True)

    cards: list[str] = []
    manifest: list[dict] = []

    for i, r in enumerate(picked):
        meta = r.get("metadata") or {}
        msgs = r["messages"]
        gt = msgs[1].get("content", "")
        if not isinstance(gt, str):
            gt = str(gt)
        src = Path(r["images"][0])
        point = meta.get("point")
        bbox = meta.get("bbox")
        bboxes = meta.get("bboxes")
        w = int(meta.get("image_w") or 0)
        h = int(meta.get("image_h") or 0)
        if (not w or not h) and src.is_file():
            with Image.open(src) as im:
                w, h = im.size
        ar = _area_ratio(bbox if isinstance(bbox, list) else None, w, h)
        ar_s = f"{ar:.4f}" if ar is not None else "n/a"
        is_neg = bool(meta.get("is_negative")) or not gt.strip()

        name = f"{i:03d}_{meta.get('sample_id', i)}".replace(":", "_").replace("/", "_")
        name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:120]
        overlay_path = img_dir / f"{name}__viz.jpg"
        _overlay(
            src,
            overlay_path,
            point if isinstance(point, list) else None,
            bbox if isinstance(bbox, list) else None,
            show_bbox=args.show_bbox,
            show_point=args.show_point,
            bboxes=bboxes if isinstance(bboxes, list) else None,
        )
        if args.copy_original:
            shutil.copy2(src, img_dir / f"{name}__raw.jpg")

        rel = overlay_path.relative_to(args.out).as_posix()
        if gt.strip():
            gt_show = gt
            gt_label = "目标文本块（模型应输出）"
            gt_class = "gt"
        else:
            gt_show = "（空字符串 — 负例 / 应弃权）"
            gt_label = "目标输出"
            gt_class = "gt empty"

        cards.append(
            f"""
<section class="card" id="s{i}">
  <div class="meta">
    <div><b>#{i}</b> <code>{html.escape(str(meta.get('sample_id','')))}</code>
      <span class="pill {'neg' if is_neg else 'pos'}">{'负例' if is_neg else '正例'}</span>
      <span class="pill">chars={len(gt)}</span>
      <span class="pill">area={html.escape(ar_s)}</span>
      {f'<span class="pill">{html.escape(str(meta.get("pool_id")))}</span>' if meta.get("pool_id") else ""}
      {f'<span class="pill">{html.escape(str(meta.get("a2_slice")))}</span>' if meta.get("a2_slice") else ""}
      {f'<span class="pill">{html.escape(str(meta.get("a2_kind")))}</span>' if meta.get("a2_kind") else ""}
      {f'<span class="pill">role={html.escape(str(meta.get("group_role")))}</span>' if meta.get("group_role") else ""}
    </div>
    <div>page=<code>{html.escape(str(meta.get('page_id','')))}</code>
        block=<code>{html.escape(str(meta.get('block_id','')))}</code>
        region=<code>{html.escape(str(meta.get('region','')))}</code></div>
  </div>
  <div class="split">
    <div class="pane img-pane">
      <div class="pane-title">训练同款截图（品红准星）</div>
      <a href="{html.escape(rel)}" target="_blank" rel="noopener">
        <img src="{html.escape(rel)}" alt="sample {i}" loading="lazy" />
      </a>
    </div>
    <div class="pane gt-pane">
      <div class="pane-title">{gt_label}</div>
      <pre class="{gt_class}">{html.escape(gt_show)}</pre>
      <label class="fb">反馈
        <select class="verdict" name="verdict_{i}" data-sid="{html.escape(str(meta.get('sample_id','')))}" data-idx="{i}">
          <option value="">(未选)</option>
          <option value="ok">符合设想</option>
          <option value="bad_point">准星/点位有问题</option>
          <option value="bad_gt">标注文本不对</option>
          <option value="too_sparse">画面太空</option>
          <option value="too_dense">过密/乱</option>
          <option value="too_hard">对本阶段过难</option>
          <option value="other">其他</option>
        </select>
      </label>
      <textarea class="note" name="note_{i}" data-sid="{html.escape(str(meta.get('sample_id','')))}" data-idx="{i}" rows="3" placeholder="备注（可选）"></textarea>
    </div>
  </div>
</section>
"""
        )
        manifest.append(
            {
                "index": i,
                "sample_id": meta.get("sample_id"),
                "image": str(src),
                "viz": str(overlay_path),
                "target": gt,
                "meta": meta,
                "area_ratio": ar,
            }
        )

    page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>POINT sample review — {html.escape(args.out.name)}</title>
  <style>
    :root {{
      --bg: #0f1115; --card: #171a21; --line: #2a3140; --text: #e8eaed;
      --muted: #9aa3b2; --gt: #c8f0c8; --accent: #7eb6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--text); }}
    header {{
      padding: 14px 20px; background: #141820; border-bottom: 1px solid var(--line);
      position: sticky; top: 0; z-index: 10;
    }}
    header .hint {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    main {{ display: flex; flex-direction: column; gap: 20px; padding: 16px 18px 40px; }}
    .card {{
      background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px;
    }}
    .meta {{ font-size: 13px; color: var(--muted); margin-bottom: 10px; line-height: 1.5; }}
    .pill {{
      display: inline-block; margin-left: 6px; padding: 1px 8px; border-radius: 999px;
      border: 1px solid var(--line); font-size: 12px; color: var(--muted);
    }}
    .pill.pos {{ color: #9fe6a8; border-color: #2f5d3a; }}
    .pill.neg {{ color: #f0c0c0; border-color: #6a3a3a; }}
    .split {{
      display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.9fr);
      gap: 14px; align-items: start;
    }}
    @media (max-width: 960px) {{
      .split {{ grid-template-columns: 1fr; }}
      .gt-pane {{ position: static !important; }}
    }}
    .pane-title {{
      font-size: 12px; letter-spacing: .04em; text-transform: uppercase;
      color: var(--muted); margin-bottom: 8px;
    }}
    .img-pane img {{
      width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px;
      background: #000; display: block;
    }}
    .gt-pane {{
      position: sticky; top: 72px;
      background: #12151c; border: 1px solid var(--line); border-radius: 8px; padding: 12px;
      max-height: calc(100vh - 90px); overflow: auto;
    }}
    .gt {{
      white-space: pre-wrap; word-break: break-word;
      background: #0b0d12; border: 1px solid #2c3b2c;
      padding: 12px; font-size: 14px; line-height: 1.5; color: var(--gt);
      min-height: 120px; margin: 0 0 12px;
    }}
    .gt.empty {{ color: #f0c0c0; border-color: #4a3030; }}
    textarea, select {{
      width: 100%; margin-top: 6px; background: #0b0d12; color: var(--text);
      border: 1px solid var(--line); border-radius: 4px; padding: 8px; font: inherit;
    }}
    code {{ color: var(--accent); }}
    .fb {{ display: block; font-size: 13px; color: var(--muted); }}
    .actions {{ margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .actions button {{
      background: #243044; color: var(--text); border: 1px solid var(--line);
      border-radius: 6px; padding: 8px 12px; cursor: pointer; font: inherit;
    }}
    .actions button.primary {{ background: #2a4a7a; border-color: #3d6aa8; }}
    .actions .status {{ font-size: 13px; color: var(--muted); }}
  </style>
</head>
<body>
<header>
  <div><b>POINT 抽检</b> — {html.escape(str(args.jsonl))} → n={n} seed={args.seed}</div>
    <div class="hint">左侧截图 · 右侧标注。绿框仅抽检（不进训练）：准星应落在绿框内。选择/备注点「导出反馈 JSON」后我才能看到。</div>
  <div class="actions">
    <button type="button" class="primary" id="btn-export">导出反馈 JSON</button>
    <button type="button" id="btn-clear">清空本页反馈</button>
    <span class="status" id="save-status"></span>
  </div>
</header>
<main>
{''.join(cards)}
</main>
<script>
(function () {{
  const KEY = "point_ocr_review::{html.escape(args.out.name)}::{html.escape(str(args.jsonl))}";

  function collect() {{
    const items = [];
    document.querySelectorAll("section.card").forEach((card) => {{
      const sel = card.querySelector("select.verdict");
      const ta = card.querySelector("textarea.note");
      if (!sel) return;
      const verdict = sel.value || "";
      const note = (ta && ta.value) || "";
      if (!verdict && !note.trim()) return;
      items.push({{
        index: Number(sel.dataset.idx),
        sample_id: sel.dataset.sid || "",
        verdict,
        note: note.trim(),
      }});
    }});
    return {{
      gallery: {json.dumps(args.out.name)},
      jsonl: {json.dumps(str(args.jsonl))},
      exported_at: new Date().toISOString(),
      n_feedback: items.length,
      items,
    }};
  }}

  function persist() {{
    const data = collect();
    try {{
      localStorage.setItem(KEY, JSON.stringify(data));
      const el = document.getElementById("save-status");
      if (el) el.textContent = "已自动保存到本机 · 已填 " + data.n_feedback + " 条";
    }} catch (e) {{
      console.warn(e);
    }}
  }}

  function restore() {{
    let raw;
    try {{ raw = localStorage.getItem(KEY); }} catch (e) {{ return; }}
    if (!raw) return;
    let data;
    try {{ data = JSON.parse(raw); }} catch (e) {{ return; }}
    const byIdx = Object.fromEntries((data.items || []).map((x) => [String(x.index), x]));
    document.querySelectorAll("select.verdict").forEach((sel) => {{
      const row = byIdx[String(sel.dataset.idx)];
      if (!row) return;
      sel.value = row.verdict || "";
      const ta = sel.closest(".gt-pane")?.querySelector("textarea.note");
      if (ta) ta.value = row.note || "";
    }});
    const el = document.getElementById("save-status");
    if (el) el.textContent = "已恢复本机保存 · " + (data.n_feedback || 0) + " 条";
  }}

  document.querySelectorAll("select.verdict, textarea.note").forEach((el) => {{
    el.addEventListener("change", persist);
    el.addEventListener("input", persist);
  }});

  document.getElementById("btn-export").addEventListener("click", () => {{
    persist();
    const data = collect();
    const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: "application/json" }});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "review_feedback.json";
    a.click();
    URL.revokeObjectURL(a.href);
    const el = document.getElementById("save-status");
    if (el) el.textContent = "已下载 review_feedback.json · 请发到聊天或放到仓库 data/review_galleries/";
  }});

  document.getElementById("btn-clear").addEventListener("click", () => {{
    if (!confirm("清空本页所有反馈？")) return;
    document.querySelectorAll("select.verdict").forEach((s) => {{ s.value = ""; }});
    document.querySelectorAll("textarea.note").forEach((t) => {{ t.value = ""; }});
    try {{ localStorage.removeItem(KEY); }} catch (e) {{}}
    const el = document.getElementById("save-status");
    if (el) el.textContent = "已清空";
  }});

  restore();
}})();
</script>
</body>
</html>
"""
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(page, encoding="utf-8")
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Gallery: {args.out / 'index.html'}")
    print(f"Manifest: {args.out / 'manifest.json'} ({n} samples)")
    print("Tip: 在页面点「导出反馈 JSON」，把文件放到仓库或直接发我。")


if __name__ == "__main__":
    main()

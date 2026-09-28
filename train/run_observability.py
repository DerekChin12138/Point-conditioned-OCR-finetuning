"""Run-directory layout, config dump, loss curves, post-train generate metrics."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.filter_qa import bucket_for_block
from point_ocr.grpo_data import val_slice
from point_ocr.infer import (
    DEFAULT_POINT_MAX_NEW_TOKENS,
    generate_point_batch,
    generate_point_text,
)
from point_ocr.metrics import EvalExample, evaluate_by_bucket, evaluate_examples
from point_ocr.prompts import POINT_PROMPT


def make_run_id(started_at: datetime | None = None, *, suffix: str = "stage_a") -> str:
    ts = (started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{suffix}"


def resolve_run_dir(
    *,
    out: Path | None,
    out_root: Path,
    run_id: str | None,
    started_at: datetime,
) -> Path:
    """Prefer explicit --out; else out_root / <timestamp>_stage_a."""
    if out is not None:
        return out
    rid = run_id or make_run_id(started_at)
    return out_root / rid


def git_rev() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    vers: dict[str, str] = {}
    for name in ("transformers", "peft", "trl", "torch", "unsloth", "huggingface_hub"):
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            vers[name] = "not_imported"
    return vers


def write_run_config(run_dir: Path, payload: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_loss_history(run_dir: Path, log_history: list[dict[str, Any]]) -> Path:
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "loss_history.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in log_history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# One PNG per metric (user-selected TRL GRPO logs). Key → filename stem.
GRPO_CURVE_METRICS: tuple[tuple[str, str], ...] = (
    ("reward", "reward"),
    ("reward_std", "reward_std"),
    ("frac_reward_zero_std", "frac_reward_zero_std"),
    ("kl", "kl"),
    ("entropy", "entropy"),
    ("clip_ratio/region_mean", "clip_ratio_region_mean"),
    ("grad_norm", "grad_norm"),
    ("learning_rate", "learning_rate"),
)


def series_from_log_history(
    log_history: list[dict[str, Any]], key: str
) -> tuple[list[int], list[float]]:
    """(step, value) for a TRL log key; drop missing/NaN."""
    steps: list[int] = []
    vals: list[float] = []
    for row in log_history:
        step = row.get("step")
        if step is None or key not in row:
            continue
        raw = row[key]
        if raw is None:
            continue
        try:
            fv = float(raw)
        except (TypeError, ValueError):
            continue
        if fv != fv:  # NaN
            continue
        steps.append(int(step))
        vals.append(fv)
    return steps, vals


def plot_grpo_metric_curves(run_dir: Path, log_history: list[dict[str, Any]]) -> list[Path]:
    """Write metrics/curves/<key>.png for each selected GRPO metric that has data."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skip GRPO curve pngs", file=sys.stderr)
        return []

    out_dir = Path(run_dir) / "metrics" / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index: dict[str, Any] = {"updated_at": datetime.now().astimezone().isoformat(), "files": {}}
    for key, stem in GRPO_CURVE_METRICS:
        steps, vals = series_from_log_history(log_history, key)
        index["files"][key] = {"n": len(vals), "file": f"{stem}.png" if steps else None}
        if not steps:
            continue
        out_path = out_dir / f"{stem}.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, vals, linewidth=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        written.append(out_path)
    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return written


def plot_loss_curves(run_dir: Path, log_history: list[dict[str, Any]]) -> Path | None:
    """Save train/eval loss curves; return path or None if matplotlib missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skip loss curves png", file=sys.stderr)
        return None

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        if "loss" in row and "eval_loss" not in row:
            train_steps.append(step)
            train_loss.append(row["loss"])
        if "eval_loss" in row:
            eval_steps.append(step)
            eval_loss.append(row["eval_loss"])

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "loss_curves.png"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if train_steps:
        ax.plot(train_steps, train_loss, label="train/loss", linewidth=1.5)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="eval/loss", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Train / Eval Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def doc_type_from_page_id(page_id: str) -> str:
    """01_article_twocol__r1 → 01_article_twocol (template / document family)."""
    if not page_id:
        return "unknown"
    return page_id.split("__", 1)[0]


def plot_final_metric_bars(run_dir: Path, report: dict[str, Any]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    by_doc = report.get("by_doc_type") or {}
    if not by_doc:
        return None

    labels = list(by_doc.keys())
    hits = [by_doc[k]["block_hit_rate"] for k in labels]
    dists = [by_doc[k]["mean_normalized_edit_distance"] for k in labels]
    ns = [by_doc[k]["n"] for k in labels]

    metrics_dir = run_dir / "metrics"
    out_path = metrics_dir / "final_metrics_by_doc_type.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(labels))
    axes[0].bar(x, hits)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("block_hit_rate by doc type")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, dists)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(
        [f"{lab}\n(n={n})" for lab, n in zip(labels, ns)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("mean normalized edit distance by doc type")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _select_eval_rows(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    """Cap generate-eval. Stratify by pool_id so a prefix is not all one pool."""
    if not max_samples or max_samples <= 0 or len(rows) <= max_samples:
        return rows
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        meta = row.get("metadata") or {}
        key = str(meta.get("pool_id") or meta.get("a2_slice") or meta.get("a1_slice") or "_")
        if key not in buckets:
            order.append(key)
        buckets[key].append(row)
    picked: list[dict[str, Any]] = []
    idxs = {k: 0 for k in order}
    while len(picked) < max_samples:
        progressed = False
        for k in order:
            i = idxs[k]
            bucket = buckets[k]
            if i >= len(bucket):
                continue
            picked.append(bucket[i])
            idxs[k] = i + 1
            progressed = True
            if len(picked) >= max_samples:
                break
        if not progressed:
            break
    return picked


def run_final_generate_eval(
    *,
    model,
    tokenizer,
    jsonl_path: Path,
    run_dir: Path,
    max_samples: int = 0,
    hit_threshold: float = 0.85,
    max_image_side: int = 0,
    max_pixels: int = 2880 * 2880,
    min_pixels: int = 448 * 448,
    max_new_tokens: int = DEFAULT_POINT_MAX_NEW_TOKENS,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Generate on held-out JSONL with the finished model; write metrics/.

    Generation is **batched** (``batch_size``, default 8).  Batch-1 greedy decode
    is latency-bound and left an 8GB GPU ~75% idle; batching 8 similar-sized
    images is ~4x faster.  Images are sorted by area inside the generator.
    """
    from PIL import Image
    from unsloth import FastVisionModel

    from point_ocr.image_resize import resize_for_ovis

    FastVisionModel.for_inference(model)

    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_samples and max_samples > 0:
        n_all = len(rows)
        rows = _select_eval_rows(rows, max_samples)
        print(
            f"post-eval generate on {len(rows)}/{n_all} rows from {jsonl_path} "
            f"(batched bs={batch_size}, max_pixels={max_pixels})",
            flush=True,
        )
    else:
        print(
            f"post-eval generate on {len(rows)} rows from {jsonl_path} "
            f"(batched bs={batch_size}, max_pixels={max_pixels})",
            flush=True,
        )

    examples: list[EvalExample] = []
    doc_types: list[str] = []
    content_buckets: list[str] = []
    scenes: list[str] = []
    pred_rows: list[dict[str, Any]] = []

    # Pass 1: build the work list and read image *headers* only (cheap) so we can
    # sort by area.  Images are decoded/resized per chunk below, so host RAM holds
    # at most ~2 chunks instead of the whole split.
    items: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        meta = row.get("metadata") or {}
        images = row.get("images") or []
        msgs = row.get("messages") or []
        # No image => text-only sample (e.g. translation). Kept, not skipped.
        if len(msgs) < 2:
            continue
        img_path = Path(images[0]) if images else None
        if img_path is not None and not img_path.exists():
            print(f"[skip] missing image: {img_path}", file=sys.stderr)
            continue
        gt = msgs[1].get("content", "")
        if not isinstance(gt, str):
            gt = str(gt)
        user_raw = msgs[0].get("content", "")
        if isinstance(user_raw, str) and user_raw.startswith("<image>"):
            prompt = user_raw[len("<image>") :] or POINT_PROMPT
        elif isinstance(user_raw, str) and user_raw.strip():
            prompt = user_raw
        else:
            prompt = POINT_PROMPT
        if img_path is not None:
            try:
                with Image.open(img_path) as probe:
                    w, h = probe.size
            except Exception as e:
                print(f"[skip] unreadable image {img_path}: {e}", file=sys.stderr)
                continue
        else:
            w, h = 0, 0
        items.append(
            {
                "row": row,
                "meta": meta,
                "img_path": img_path,
                "area": int(w) * int(h),
                "gt": gt,
                "prompt": prompt if prompt.strip() else None,
            }
        )

    # Sort by area so a batch pads as little as possible.
    items.sort(key=lambda it: it["area"])

    def _load_chunk(chunk: list[dict[str, Any]]):
        out = []
        for it in chunk:
            if it["img_path"] is None:  # text-only sample
                out.append(None)
                continue
            img = Image.open(it["img_path"]).convert("RGB")
            if max_pixels and max_pixels > 0:
                img = resize_for_ovis(img, min_pixels=min_pixels or (448 * 448), max_pixels=max_pixels)
            elif max_image_side and max(img.size) > max_image_side:
                img.thumbnail((max_image_side, max_image_side))
            out.append(img)
        return out

    n_items = len(items)
    bs = max(1, int(batch_size))
    t0 = time.time()
    state = {"bs": bs}

    def _gen(chunk: list[dict[str, Any]], imgs: list[Any]):
        """Generate one chunk; on failure (usually OOM) halve the batch **for good**."""
        try:
            outs = generate_point_batch(
                model,
                tokenizer,
                imgs,
                [it["prompt"] for it in chunk],
                max_new_tokens=max_new_tokens,
                batch_size=len(chunk),
                sort_by_area=True,
                clean=True,
            )
            return [(o.raw, o.cleaned, o.format_leak) for o in outs]
        except Exception as e:
            if len(chunk) == 1:
                try:
                    out = generate_point_text(
                        model, tokenizer, imgs[0], chunk[0]["prompt"], max_new_tokens=max_new_tokens, clean=True
                    )
                    return [(out.raw, out.cleaned, out.format_leak)]
                except Exception as e2:
                    print(f"[infer error] {chunk[0]['meta'].get('sample_id')}: {e2}", file=sys.stderr)
                    return [("", "", False)]
            mid = len(chunk) // 2
            if mid < state["bs"]:
                print(
                    f"[infer batch error] {e}; reducing batch size {state['bs']} → {mid} for the rest of the run",
                    file=sys.stderr,
                )
                state["bs"] = mid
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return _gen(chunk[:mid], imgs[:mid]) + _gen(chunk[mid:], imgs[mid:])

    # Prefetch the next chunk's images on a CPU thread while the GPU generates.
    executor = ThreadPoolExecutor(max_workers=1)
    start = 0
    pending = executor.submit(_load_chunk, items[0 : state["bs"]]) if n_items else None
    try:
        while start < n_items:
            cur_bs = max(1, int(state["bs"]))
            chunk = items[start : start + cur_bs]
            imgs = pending.result() if pending is not None else _load_chunk(chunk)
            nxt = start + cur_bs
            pending = executor.submit(_load_chunk, items[nxt : nxt + cur_bs]) if nxt < n_items else None
            preds = _gen(chunk, imgs)
            start += len(chunk)

            for it, (raw_pred, pred, leaked) in zip(chunk, preds):
                meta, gt, img_path = it["meta"], it["gt"], it["img_path"]
                sid = str(meta.get("sample_id") or len(examples))
                is_neg = bool(meta.get("is_negative")) or not gt.strip()
                examples.append(EvalExample(sid, pred, gt, is_neg, raw_prediction=raw_pred))
                doc_types.append(doc_type_from_page_id(str(meta.get("page_id") or "")))
                content_buckets.append(bucket_for_block(gt))
                scenes.append(str(val_slice(it["row"]) or meta.get("grpo_scene") or "regular"))
                pred_rows.append(
                    {
                        "sample_id": sid,
                        "doc_type": doc_types[-1],
                        "content_bucket": content_buckets[-1],
                        "grpo_scene": scenes[-1],
                        "is_negative": is_neg,
                        "target": gt,
                        "prediction": pred,
                        "prediction_raw": raw_pred,
                        "format_leak": leaked,
                        "image": str(img_path),
                    }
                )
            done = min(start, n_items)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_items - done) / rate if rate > 0 else 0
            print(
                f"post-eval {done}/{n_items}  (bs={state['bs']})  {rate:.2f} samp/s  eta={eta:.0f}s",
                flush=True,
            )
    finally:
        executor.shutdown(wait=False)

    overall = evaluate_examples(examples, hit_threshold=hit_threshold)
    by_doc = evaluate_by_bucket(examples, doc_types, hit_threshold=hit_threshold)
    by_content = evaluate_by_bucket(examples, content_buckets, hit_threshold=hit_threshold)
    by_scene = evaluate_by_bucket(examples, scenes, hit_threshold=hit_threshold)

    report = {
        "split": str(jsonl_path),
        "n_scored": len(examples),
        "hit_threshold": hit_threshold,
        "overall": overall.to_dict(),
        "by_doc_type": {k: v.to_dict() for k, v in by_doc.items()},
        "by_content_bucket": {k: v.to_dict() for k, v in by_content.items()},
        "by_grpo_scene": {k: v.to_dict() for k, v in by_scene.items()},
    }

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pred_path = metrics_dir / "final_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    bar = plot_final_metric_bars(run_dir, report)
    if bar:
        report["plots"] = {"by_doc_type": str(bar)}
        (metrics_dir / "final_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("final metrics:", json.dumps(overall.to_dict(), indent=2))
    return report

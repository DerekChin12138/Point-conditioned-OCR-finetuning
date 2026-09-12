"""Run-directory layout, config dump, loss curves, post-train generate metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.filter_qa import bucket_for_block
from point_ocr.infer import DEFAULT_POINT_MAX_NEW_TOKENS, generate_point_text
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


def run_final_generate_eval(
    *,
    model,
    tokenizer,
    jsonl_path: Path,
    run_dir: Path,
    max_samples: int = 0,
    hit_threshold: float = 0.85,
    max_image_side: int = 1536,
    max_new_tokens: int = DEFAULT_POINT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Generate on held-out JSONL with the finished model; write metrics/."""
    from PIL import Image
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)

    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]

    examples: list[EvalExample] = []
    doc_types: list[str] = []
    content_buckets: list[str] = []
    pred_rows: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        meta = row.get("metadata") or {}
        images = row.get("images") or []
        msgs = row.get("messages") or []
        if not images or len(msgs) < 2:
            continue
        img_path = Path(images[0])
        if not img_path.exists():
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

        img = Image.open(img_path).convert("RGB")
        if max_image_side and max(img.size) > max_image_side:
            img.thumbnail((max_image_side, max_image_side))

        try:
            out = generate_point_text(
                model,
                tokenizer,
                img,
                prompt,
                max_new_tokens=max_new_tokens,
                clean=True,
            )
            raw_pred = out.raw
            pred = out.cleaned
            leaked = out.format_leak
        except Exception as e:
            raw_pred = ""
            pred = ""
            leaked = False
            print(f"[infer error] {meta.get('sample_id', i)}: {e}", file=sys.stderr)

        sid = str(meta.get("sample_id") or i)
        is_neg = bool(meta.get("is_negative")) or not gt.strip()
        examples.append(
            EvalExample(sid, pred, gt, is_neg, raw_prediction=raw_pred)
        )
        doc_types.append(doc_type_from_page_id(str(meta.get("page_id") or "")))
        content_buckets.append(bucket_for_block(gt))
        pred_rows.append(
            {
                "sample_id": sid,
                "doc_type": doc_types[-1],
                "content_bucket": content_buckets[-1],
                "is_negative": is_neg,
                "target": gt,
                "prediction": pred,
                "prediction_raw": raw_pred,
                "format_leak": leaked,
                "image": str(img_path),
            }
        )
        if (i + 1) % 20 == 0:
            print(f"post-eval {i + 1}/{len(rows)}")

    overall = evaluate_examples(examples, hit_threshold=hit_threshold)
    by_doc = evaluate_by_bucket(examples, doc_types, hit_threshold=hit_threshold)
    by_content = evaluate_by_bucket(examples, content_buckets, hit_threshold=hit_threshold)

    report = {
        "split": str(jsonl_path),
        "n_scored": len(examples),
        "hit_threshold": hit_threshold,
        "overall": overall.to_dict(),
        "by_doc_type": {k: v.to_dict() for k, v in by_doc.items()},
        "by_content_bucket": {k: v.to_dict() for k, v in by_content.items()},
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

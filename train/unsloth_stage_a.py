#!/usr/bin/env python3
"""Stage A POINT SFT with Unsloth FastVisionModel (preferred over LLaMA-Factory).

Run on AutoDL / Linux + NVIDIA CUDA only (not macOS).

Example:
  uv run python train/unsloth_stage_a.py \\
    --data data/splits/train.jsonl \\
    --val data/splits/val.jsonl \\
    --model ATH-MaaS/OvisOCR2

Each run writes under checkpoints/<YYYYMMDD_HHMMSS>_stage_a/ by default:
  run_config.json, tb/, metrics/, checkpoint-*, adapter_final/

If OvisOCR2 fails to load under Unsloth, smoke-test first:
  uv run python train/unsloth_stage_a.py --smoke-load-only
and see docs/UNSLOTH.md for fallbacks.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "train"))

from point_ocr.prompts import POINT_PROMPT
from run_observability import (
    git_rev,
    package_versions,
    plot_loss_curves,
    resolve_run_dir,
    run_final_generate_eval,
    write_loss_history,
    write_run_config,
)


def sharegpt_to_unsloth(rows: list[dict], *, lazy_images: bool = True) -> list[dict]:
    """Convert ShareGPT JSONL → Unsloth vision conversations.

    lazy_images=True (default): keep filesystem paths instead of decoding every
    JPEG into RAM up-front. Full decode of ~4500 desktop screenshots easily
    exhausts host RAM while GPU VRAM stays low (model still 4-bit).
    """
    from PIL import Image

    out: list[dict] = []
    for row in rows:
        images = row.get("images") or []
        msgs = row.get("messages") or []
        if not images or len(msgs) < 2:
            continue
        user_raw = msgs[0].get("content", "")
        if isinstance(user_raw, str) and user_raw.startswith("<image>"):
            text = user_raw[len("<image>") :]
        elif isinstance(user_raw, str):
            text = user_raw
        else:
            text = POINT_PROMPT
        if not text.strip():
            text = POINT_PROMPT
        target = msgs[1].get("content", "")
        if not isinstance(target, str):
            target = str(target)
        img_path = Path(images[0])
        if not img_path.exists():
            print(f"[skip] missing image: {img_path}", file=sys.stderr)
            continue
        if lazy_images:
            image_payload: object = str(img_path.resolve())
        else:
            image_payload = Image.open(img_path).convert("RGB")
        out.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image", "image": image_payload},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target}],
                    },
                ]
            }
        )
    return out


class LazyImageDataset:
    """Torch-style dataset: decode JPEG only inside __getitem__."""

    def __init__(self, rows: list[dict], *, max_side: int = 0):
        from PIL import Image as _Image

        self._Image = _Image
        self.rows = sharegpt_to_unsloth(rows, lazy_images=True)
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        sample = self.rows[idx]
        messages = []
        for msg in sample["messages"]:
            content = []
            for part in msg["content"]:
                if part.get("type") == "image":
                    img = self._Image.open(part["image"]).convert("RGB")
                    if self.max_side and max(img.size) > self.max_side:
                        img.thumbnail((self.max_side, self.max_side))
                    content.append({"type": "image", "image": img})
                else:
                    content.append(dict(part))
            messages.append({"role": msg["role"], "content": content})
        return {"messages": messages}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def smoke_load(model_id: str) -> None:
    from unsloth import FastVisionModel

    print(f"Loading {model_id} with FastVisionModel ...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    print("OK: model + tokenizer loaded")
    print("type:", type(model))
    del model, tokenizer


def train(args: argparse.Namespace) -> None:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    started_at = datetime.now().astimezone()
    run_dir = resolve_run_dir(
        out=args.out,
        out_root=args.out_root,
        run_id=args.run_id,
        started_at=started_at,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.data)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    val_rows: list[dict] = []
    eval_enabled = bool(args.val) and not args.no_eval
    if eval_enabled:
        if not args.val.exists():
            raise SystemExit(f"--val not found: {args.val}")
        val_rows = load_jsonl(args.val)
        if args.max_val_samples and args.max_val_samples > 0:
            val_rows = val_rows[: args.max_val_samples]

    run_config: dict = {
        "started_at": started_at.isoformat(),
        "run_dir": str(run_dir.resolve()),
        "git_commit": git_rev(),
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_train_rows_loaded": len(rows),
        "n_val_rows_loaded": len(val_rows) if eval_enabled else 0,
        "packages": package_versions(),
        "tensorboard_logdir": str(tb_dir.resolve()),
        "notes": {
            "during_training": "train/loss + eval/loss only (TensorBoard)",
            "after_training": "generate metrics on final model → metrics/final_report.json",
        },
    }
    write_run_config(run_dir, run_config)
    print(f"run_dir = {run_dir}")
    print(f"TensorBoard logdir = {tb_dir}  (point AutoDL TensorBoard here)")

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_length,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )
    FastVisionModel.for_training(model)

    dataset = LazyImageDataset(rows, max_side=args.max_image_side)
    if len(dataset) == 0:
        raise SystemExit(f"No usable samples from {args.data}")
    print(
        f"train samples={len(dataset)} from {args.data} "
        f"(lazy images, max_side={args.max_image_side or 'none'})"
    )

    eval_dataset = None
    if eval_enabled:
        eval_dataset = LazyImageDataset(val_rows, max_side=args.max_image_side)
        if len(eval_dataset) == 0:
            raise SystemExit(f"No usable val samples from {args.val}")
        print(f"val samples={len(eval_dataset)} from {args.val}")

    sft_kwargs: dict = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        logging_dir=str(tb_dir),
        report_to=["tensorboard"],
        save_strategy="steps",
        save_steps=args.save_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        output_dir=str(run_dir),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length,
        bf16=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
    )
    if eval_dataset is not None:
        sft_kwargs.update(
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=args.eval_batch_size,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        sft_kwargs["eval_strategy"] = "no"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**sft_kwargs),
    )
    train_result = trainer.train()

    adapter_dir = run_dir / "adapter_final"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA → {adapter_dir}")

    log_history = list(trainer.state.log_history)
    write_loss_history(run_dir, log_history)
    curve = plot_loss_curves(run_dir, log_history)
    if curve:
        print(f"loss curves → {curve}")

    ended_at = datetime.now().astimezone()
    run_config.update(
        {
            "ended_at": ended_at.isoformat(),
            "duration_sec": (ended_at - started_at).total_seconds(),
            "train_result": {
                k: (float(v) if hasattr(v, "item") else v)
                for k, v in dict(train_result.metrics).items()
            },
            "best_metric": getattr(trainer.state, "best_metric", None),
            "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
            "adapter_final": str(adapter_dir.resolve()),
            "n_train_dataset": len(dataset),
            "n_eval_dataset": len(eval_dataset) if eval_dataset is not None else 0,
        }
    )
    # refresh package versions after imports
    run_config["packages"] = package_versions()
    write_run_config(run_dir, run_config)

    if not args.skip_post_eval:
        post_path = args.post_eval or args.val
        if post_path is None or not Path(post_path).exists():
            print("[warn] skip post-eval: no --post-eval / --val path", file=sys.stderr)
        else:
            print(f"post-train generate eval on {post_path} ...")
            report = run_final_generate_eval(
                model=model,
                tokenizer=tokenizer,
                jsonl_path=Path(post_path),
                run_dir=run_dir,
                max_samples=args.post_eval_max,
                hit_threshold=args.hit_threshold,
                max_image_side=args.max_image_side,
            )
            run_config["post_eval"] = {
                "path": str(post_path),
                "overall": report.get("overall"),
            }
            write_run_config(run_dir, run_config)

    print(f"done. run_dir={run_dir}")
    print(f"AutoDL TensorBoard → {tb_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unsloth Stage A POINT SFT")
    ap.add_argument("--data", type=Path, default=ROOT / "data/splits/train.jsonl")
    ap.add_argument(
        "--val",
        type=Path,
        default=ROOT / "data/splits/val.jsonl",
        help="Val JSONL for eval/loss during training (and default post-eval)",
    )
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "checkpoints",
        help="Parent dir for timestamped runs",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="Override run folder name (default: YYYYMMDD_HHMMSS_stage_a)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, use this exact run directory (skips auto timestamp under --out-root)",
    )
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 overrides epochs")
    ap.add_argument("--max-samples", type=int, default=0, help="debug subset; 0=all")
    ap.add_argument("--max-val-samples", type=int, default=0, help="cap val for eval/loss; 0=all")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument(
        "--max-image-side",
        type=int,
        default=1536,
        help="Downscale images so longest side <= this (0=keep original). Saves RAM/VRAM.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; keep 0 on small-RAM hosts to avoid multiply-forked copies.",
    )
    ap.add_argument(
        "--finetune-vision",
        action="store_true",
        help="Also LoRA vision tower (default: freeze vision)",
    )
    ap.add_argument("--no-eval", action="store_true", help="Disable mid-train eval/loss")
    ap.add_argument(
        "--post-eval",
        type=Path,
        default=None,
        help="JSONL for post-train generate metrics (default: --val)",
    )
    ap.add_argument(
        "--post-eval-max",
        type=int,
        default=0,
        help="Cap post-eval samples; 0=all (val ~250 is usually fine)",
    )
    ap.add_argument("--hit-threshold", type=float, default=0.85)
    ap.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="Skip generate metrics after training",
    )
    ap.add_argument(
        "--smoke-load-only",
        action="store_true",
        help="Only test FastVisionModel.from_pretrained(model)",
    )
    args = ap.parse_args()

    if args.smoke_load_only:
        smoke_load(args.model)
        return
    train(args)


if __name__ == "__main__":
    main()

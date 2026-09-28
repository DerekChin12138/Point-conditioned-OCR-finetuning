#!/usr/bin/env python3
"""POINT SFT with Unsloth FastVisionModel. Linux + NVIDIA only.

Current Q1 recipe and flags: see the repo README.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# Must be set before torch initialises CUDA: reduces allocator fragmentation on 8GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]

def _gc_mode():
    """`POINT_OCR_GC=standard` -> plain recompute checkpointing (no host-RAM offload).

    Unsloth's default offloads saved activations to pinned host memory, which can
    exhaust a 15GB host on ~8000-token pages (observed host OOM during GRPO).
    """
    import os as _os

    return True if _os.environ.get("POINT_OCR_GC", "").strip().lower() in {"standard", "true", "1"} else "unsloth"

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


def _quiet_unsloth_noise() -> None:
    """Drop known-benign Unsloth warnings we cannot fix without forking the zoo."""

    class _Filter(logging.Filter):
        _DROP = (
            "Failed to register input-embedding hook",
            "Falling back to pre-forward hook",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not any(s in msg for s in self._DROP)

    for name in ("unsloth_zoo", "unsloth_zoo.log", "unsloth"):
        logging.getLogger(name).addFilter(_Filter())


def _align_special_tokens(model, tokenizer) -> None:
    """Keep config / generation_config eos·pad in sync with the tokenizer (silences HF warn)."""
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    configs = [getattr(model, "config", None), getattr(model, "generation_config", None)]
    cfg = getattr(model, "config", None)
    if cfg is not None:
        configs.append(getattr(cfg, "text_config", None))
    for c in configs:
        if c is None:
            continue
        if eos is not None and hasattr(c, "eos_token_id"):
            c.eos_token_id = eos
        if pad is not None and hasattr(c, "pad_token_id"):
            c.pad_token_id = pad


def _ensure_vision_image_size(model) -> None:
    """Qwen3.5 vision_config has no image_size; Unsloth otherwise falls back to 512."""
    cfg = getattr(model, "config", None)
    vc = getattr(cfg, "vision_config", None) if cfg is not None else None
    if vc is None or getattr(vc, "image_size", None) is not None:
        return
    npe = getattr(vc, "num_position_embeddings", None)
    ps = int(getattr(vc, "patch_size", None) or 16)
    if not npe or ps <= 0:
        return
    side = int(round(math.sqrt(float(npe)))) * ps
    if side > 0:
        vc.image_size = side
        print(f"vision_config.image_size ← {side} (from num_position_embeddings={npe})", flush=True)


def estimate_vision_tokens(max_pixels: int) -> int:
    """Approx. vision tokens a Qwen3.5/VL image costs: pixels / (patch*merge)^2.

    patch_size=16, merge_size=2 → 1 token per 16*2*16*2 = 1024 pixels.
    """
    if not max_pixels or max_pixels <= 0:
        return 0
    return int(max_pixels) // 1024


def warn_vision_budget(max_pixels: int, max_seq_length: int, *, slack: int = 2048) -> None:
    """Warn when one image alone could crowd out text+answer from max_seq_length."""
    est = estimate_vision_tokens(max_pixels)
    if est and max_seq_length and est + slack > max_seq_length:
        print(
            f"[warn] --max-pixels {max_pixels} → ~{est} vision tokens; with "
            f"--max-seq-length {max_seq_length} only ~{max(0, max_seq_length - est)} "
            f"tokens remain for prompt+answer. Raise --max-seq-length or lower "
            f"--max-pixels.",
            flush=True,
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
        # `images` is optional: text-only rows (e.g. a translation-only task) are
        # kept. UnslothVisionDataCollator natively supports batches that mix
        # image samples with text-only samples.
        if len(msgs) < 2:
            continue
        user_raw = msgs[0].get("content", "")
        if isinstance(user_raw, str) and user_raw.startswith("<image>"):
            text = user_raw[len("<image>") :]
        elif isinstance(user_raw, str):
            text = user_raw
        else:
            text = ""
        if not text.strip():
            text = POINT_PROMPT
        target = msgs[1].get("content", "")
        if not isinstance(target, str):
            target = str(target)
        user_parts: list[dict] = [{"type": "text", "text": text}]
        if images:
            img_path = Path(images[0])
            if not img_path.exists():
                print(f"[skip] missing image: {img_path}", file=sys.stderr)
                continue
            if lazy_images:
                image_payload: object = str(img_path.resolve())
            else:
                image_payload = Image.open(img_path).convert("RGB")
            user_parts.append({"type": "image", "image": image_payload})
        out.append(
            {
                "messages": [
                    {"role": "user", "content": user_parts},
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

    def __init__(
        self,
        rows: list[dict],
        *,
        max_side: int = 0,
        min_pixels: int = 0,
        max_pixels: int = 0,
    ):
        from PIL import Image as _Image

        self._Image = _Image
        self.rows = sharegpt_to_unsloth(rows, lazy_images=True)
        self.max_side = max_side
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        from point_ocr.image_resize import resize_for_ovis

        sample = self.rows[idx]
        messages = []
        for msg in sample["messages"]:
            content = []
            for part in msg["content"]:
                if part.get("type") == "image":
                    img = self._Image.open(part["image"]).convert("RGB")
                    if self.max_pixels and self.max_pixels > 0:
                        img = resize_for_ovis(
                            img,
                            min_pixels=self.min_pixels or (448 * 448),
                            max_pixels=self.max_pixels,
                        )
                    elif self.max_side and max(img.size) > self.max_side:
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


def _preferred_compute_dtype():
    """bf16 on Ampere+; else fp16. Used for --no-4bit LoRA cast."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _freeze_vision_lora(model) -> int:
    """Set requires_grad=False on vision-tower LoRA tensors; return how many changed.

    Needed because Unsloth's ``finetune_vision_layers`` is a no-op for Qwen3.5
    (``target_modules="all-linear"`` always attaches ``model.visual.*`` LoRA).
    """
    vision_keys = ("vision", "visual", "image", "patch", "merger")
    n = 0
    for name, p in model.named_parameters():
        low = name.lower()
        if "lora" not in low:
            continue
        if any(k in low for k in vision_keys) and p.requires_grad:
            p.requires_grad = False
            n += 1
    return n


def _cast_trainable_lora_(model, dtype) -> int:
    """Unsloth/PEFT often creates LoRA in float32; cast adapters to compute dtype."""
    n = 0
    for name, p in model.named_parameters():
        if "lora" not in name.lower() or not p.requires_grad:
            continue
        if p.dtype != dtype:
            p.data = p.data.to(dtype=dtype)
            n += 1
    return n


def _summarize_lora_dtypes(model) -> tuple[int, dict[str, int]]:
    lora_dtypes: dict[str, int] = {}
    n_lora = 0
    for name, p in model.named_parameters():
        if "lora" not in name.lower() or not p.requires_grad:
            continue
        n_lora += 1
        key = str(p.dtype)
        lora_dtypes[key] = lora_dtypes.get(key, 0) + p.numel()
    return n_lora, lora_dtypes


def smoke_load(model_id: str) -> None:
    from unsloth import FastVisionModel

    print(f"Loading {model_id} with FastVisionModel ...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=True,
        use_gradient_checkpointing=_gc_mode(),
    )
    print("OK: model + tokenizer loaded")
    print("type:", type(model))
    del model, tokenizer


def _tune_unsloth_ce_env() -> None:
    """WSL/Windows: Unsloth fused CE sizes chunks from cuda mem_get_info() leftover.

    After a vision forward that leftover is often ~0 even when Task Manager shows
    lots of *shared* GPU memory (that is host RAM, not CUDA device free). The
    check then raises ``No or negligible GPU memory available for fused cross
    entropy``. Force a fixed chunk count so CE never consults free VRAM.
    Must run before ``import unsloth`` — zoo reads the env at import time.
    """
    if "UNSLOTH_CE_LOSS_N_CHUNKS" not in os.environ:
        os.environ["UNSLOTH_CE_LOSS_N_CHUNKS"] = "32"


def train(args: argparse.Namespace) -> None:
    _tune_unsloth_ce_env()
    _quiet_unsloth_noise()
    if getattr(args, "tf32", False):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:  # newer API; harmless if absent
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            pass
    # Avoid Unsloth auto-picking a high num_proc then clamping with a warning.
    if "UNSLOTH_DATASET_NUM_PROC" not in os.environ:
        # Lazy vision collator does little CPU map work; keep this small.
        os.environ["UNSLOTH_DATASET_NUM_PROC"] = "2"

    print(
        f"UNSLOTH_CE_LOSS_N_CHUNKS={os.environ.get('UNSLOTH_CE_LOSS_N_CHUNKS')}",
        flush=True,
    )

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    # Zoo configures its loggers at import time; re-attach after import.
    _quiet_unsloth_noise()

    started_at = datetime.now().astimezone()
    run_dir = resolve_run_dir(
        out=args.out,
        out_root=args.out_root,
        run_id=args.run_id,
        started_at=started_at,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir.resolve()
    tb_dir = (run_dir / "tb").resolve()
    tb_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    # transformers ≥5.2: prefer env over deprecated TrainingArguments.logging_dir
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)

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

    load_kwargs: dict = dict(
        load_in_4bit=not args.no_4bit,
        use_gradient_checkpointing=_gc_mode(),
        max_seq_length=args.max_seq_length,
    )
    if args.no_4bit:
        load_kwargs["dtype"] = _preferred_compute_dtype()
    model, tokenizer = FastVisionModel.from_pretrained(args.model, **load_kwargs)
    _align_special_tokens(model, tokenizer)
    _ensure_vision_image_size(model)
    print(f"base load_in_4bit={not args.no_4bit} (bf16 LoRA path when False)")
    if args.no_4bit:
        base_dtypes: dict[str, int] = {}
        for p in model.parameters():
            key = str(p.dtype)
            base_dtypes[key] = base_dtypes.get(key, 0) + p.numel()
        print(f"base param dtypes (numel): {base_dtypes}")

    if args.adapter:
        from peft import PeftModel

        adapter = args.adapter.resolve()
        cfg_path = adapter / "adapter_config.json"
        if not adapter.is_dir() or not cfg_path.is_file():
            raise SystemExit(
                f"--adapter is not a PEFT dir (need adapter_config.json): {adapter}"
            )
        weight_ok = any(
            (adapter / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt")
        )
        if not weight_ok:
            raise SystemExit(f"--adapter has config but no LoRA weights: {adapter}")
        peft_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        print(
            f"continue-train from adapter={adapter}\n"
            f"  peft_type={peft_cfg.get('peft_type')} r={peft_cfg.get('r')} "
            f"lora_alpha={peft_cfg.get('lora_alpha')} "
            f"target_modules={peft_cfg.get('target_modules')}",
            flush=True,
        )
        print(
            "[note] --lora-rank/--lora-alpha ignored with --adapter; "
            "weights and rank come from the PEFT dir above.",
            flush=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        loaded_from = getattr(model, "peft_config", None)
        if loaded_from:
            active = next(iter(loaded_from.values())) if isinstance(loaded_from, dict) else loaded_from
            print(
                f"  PeftModel active r={getattr(active, 'r', None)} "
                f"lora_alpha={getattr(active, 'lora_alpha', None)}",
                flush=True,
            )
        run_config["loaded_adapter"] = {
            "path": str(adapter),
            "r": peft_cfg.get("r"),
            "lora_alpha": peft_cfg.get("lora_alpha"),
            "peft_type": peft_cfg.get("peft_type"),
        }
        write_run_config(run_dir, run_config)
    else:
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

    # Unsloth's `finetune_vision_layers` is a no-op for Qwen3.5: with
    # target_modules="all-linear" the vision tower (model.visual.*) always gets a
    # LoRA, and it is always trainable.  Enforce the flag ourselves so
    # --finetune-vision actually decides whether the vision tower learns.
    # (Verified: True/False gave identical trainable param counts before this.)
    if not args.finetune_vision:
        n_freeze = _freeze_vision_lora(model)
        print(
            f"froze {n_freeze} vision LoRA tensors "
            f"(--finetune-vision not set; Unsloth flag is a no-op for Qwen3.5)",
            flush=True,
        )
    FastVisionModel.for_training(model)
    _align_special_tokens(model, tokenizer)

    if args.no_4bit:
        # Unsloth "16bit LoRA" keeps adapter weights in fp32 by default (mixed precision).
        # Curriculum wants bf16 adapters matching the bf16 base — cast explicitly.
        target = _preferred_compute_dtype()
        n_cast = _cast_trainable_lora_(model, target)
        if n_cast:
            print(f"cast {n_cast} trainable LoRA tensors → {target}", flush=True)
        n_lora, lora_dtypes = _summarize_lora_dtypes(model)
        print(f"trainable LoRA tensors={n_lora} dtypes (numel): {lora_dtypes}")
        ok = any(k in {"torch.bfloat16", "torch.float16"} for k in lora_dtypes)
        bad = any("uint" in k or "int4" in k or "int8" in k for k in lora_dtypes)
        if bad or not ok:
            raise SystemExit(
                f"--no-4bit requested but LoRA dtypes look wrong: {lora_dtypes}. Abort."
            )
        if "torch.bfloat16" not in lora_dtypes:
            print("[warn] LoRA is fp16, not bf16 — check GPU bf16 support", flush=True)

    dataset = LazyImageDataset(
        rows,
        max_side=args.max_image_side,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No usable samples from {args.data}")
    if args.max_pixels and args.max_pixels > 0:
        print(
            f"train samples={len(dataset)} from {args.data} "
            f"(lazy images, min_pixels={args.min_pixels}, max_pixels={args.max_pixels}, "
            f"collator_resize={args.collator_resize}, "
            f"~{estimate_vision_tokens(args.max_pixels)} vision tokens/image)"
        )
    else:
        print(
            f"train samples={len(dataset)} from {args.data} "
            f"(lazy images, max_side={args.max_image_side or 'none'}, "
            f"collator_resize={args.collator_resize})"
        )
    warn_vision_budget(args.max_pixels, args.max_seq_length)

    eval_dataset = None
    if eval_enabled:
        eval_dataset = LazyImageDataset(
            val_rows,
            max_side=args.max_image_side,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        if len(eval_dataset) == 0:
            raise SystemExit(f"No usable val samples from {args.val}")
        print(f"val samples={len(eval_dataset)} from {args.val}")

    # HF ≥5.2: warmup_ratio deprecated; float warmup_steps < 1 still means a fraction.
    warmup_steps = float(args.warmup_ratio)
    sft_kwargs: dict = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        save_strategy="steps",
        save_steps=args.save_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        output_dir=str(run_dir),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length,
        bf16=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=bool(args.pin_memory and args.num_workers > 0),
        dataloader_persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        dataloader_prefetch_factor=(int(args.prefetch_factor) if args.num_workers > 0 else None),
    )
    if eval_dataset is not None:
        # load_best_model_at_end requires save_steps % eval_steps == 0
        if args.save_steps % args.eval_steps != 0:
            aligned = ((args.save_steps + args.eval_steps - 1) // args.eval_steps) * args.eval_steps
            print(
                f"[warn] save_steps={args.save_steps} is not a multiple of "
                f"eval_steps={args.eval_steps}; aligning save_steps → {aligned}",
                flush=True,
            )
            args.save_steps = aligned
            sft_kwargs["save_steps"] = aligned
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

    sft_args = SFTConfig(**sft_kwargs)
    sft_args.report_to = ["tensorboard"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer, resize=args.collator_resize),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
    )
    if args.early_stopping_patience and args.early_stopping_patience > 0 and eval_dataset is not None:
        from transformers import EarlyStoppingCallback

        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=int(args.early_stopping_patience))
        )
        print(f"early_stopping_patience={args.early_stopping_patience}")
    print(f"effective TensorBoard logging_dir = {os.environ.get('TENSORBOARD_LOGGING_DIR')}")
    resume = getattr(args, "resume_from_checkpoint", None)
    if resume:
        resume = Path(resume).resolve()
        if not (resume / "trainer_state.json").is_file():
            raise SystemExit(f"--resume-from-checkpoint is not a Trainer checkpoint: {resume}")
        print(f"resuming from checkpoint {resume}", flush=True)
        run_config["resumed_from"] = str(resume)
        write_run_config(run_dir, run_config)
    train_result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)

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
                max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
                batch_size=args.post_eval_batch_size,
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
        "--adapter",
        type=Path,
        default=None,
        help="Continue-train from an existing LoRA dir (e.g. A1 adapter_final). "
        "Keeps that adapter's rank/α; --lora-rank/--lora-alpha ignored when set.",
    )
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
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 overrides epochs")
    ap.add_argument("--max-samples", type=int, default=0, help="debug subset; 0=all")
    ap.add_argument("--max-val-samples", type=int, default=0, help="cap val for eval/loss; 0=all")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Must be a multiple of --eval-steps when load_best_model_at_end is on.",
    )
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup as a fraction of total steps (passed to HF warmup_steps as float < 1).",
    )
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lr-scheduler", default="cosine")
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping; 0 disables. HF default is 1.0.",
    )
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=2048 * 2048,
        help=(
            "Max total pixels (smart resize). Vision tokens ≈ max_pixels/1024 "
            "(patch16×merge2); keep max_pixels/1024 + ~2048 <= --max-seq-length. "
            "2048²≈4096 tokens (default); 2880²≈8100 needs seq>=12288. 0 disables."
        ),
    )
    ap.add_argument(
        "--collator-resize",
        choices=("max", "min"),
        default="max",
        help=(
            "UnslothVisionDataCollator resize mode. 'max' (default) leaves the "
            "image as prepared by --max-pixels; 'min' caps the longer side at "
            "vision_config.image_size (Qwen3.5-0.8B → 768), which silently "
            "undoes --max-pixels. Use 'min' only to reproduce pre-2026-09-28 runs."
        ),
    )
    ap.add_argument(
        "--min-pixels",
        type=int,
        default=448 * 448,
        help="OvisOCR2-style min total pixels (smart resize).",
    )
    ap.add_argument(
        "--max-image-side",
        type=int,
        default=0,
        help="Legacy longest-side thumbnail; ignored when --max-pixels > 0. 0=off.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers for image decode/resize. Host has 32 cores; "
        "8 keeps the GPU fed without flooding 15GB RAM. 0 = main process.",
    )
    ap.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin host batches for faster H2D copies (needs --num-workers>0).",
    )
    ap.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep worker processes alive between epochs (needs --num-workers>0).",
    )
    ap.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches prefetched per worker (needs --num-workers>0).",
    )
    ap.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 matmul/cudnn on Ampere+ (faster, negligible quality loss).",
    )
    ap.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load base model in bf16 (not QLoRA 4bit). Recommended for A1_v2/A2_v2.",
    )
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="HF EarlyStoppingCallback patience on eval_loss; 0=disabled. Use 3–4 for A1/A2 v2.",
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
    ap.add_argument(
        "--post-eval-batch-size",
        type=int,
        default=8,
        help="Batched generate size for post-eval (1 = old sequential). "
        "8 is ~4x faster on an 8GB GPU; lower it if a large-image batch OOMs.",
    )
    ap.add_argument("--hit-threshold", type=float, default=0.85)
    ap.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="Skip generate metrics after training",
    )
    ap.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume an interrupted run from a Trainer checkpoint dir "
        "(restores LoRA weights + optimizer/scheduler + RNG). Use with the same --run-id.",
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

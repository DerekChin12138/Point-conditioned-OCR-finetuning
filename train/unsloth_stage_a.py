#!/usr/bin/env python3
"""Stage A POINT SFT with Unsloth FastVisionModel (preferred over LLaMA-Factory).

Run on AutoDL / Linux + NVIDIA CUDA only (not macOS).

Example:
  uv run python train/unsloth_stage_a.py \\
    --data data/processed/synth/point_sharegpt.jsonl \\
    --model ATH-MaaS/OvisOCR2 \\
    --out checkpoints/stage_a_point_unsloth \\
    --max-steps 200

If OvisOCR2 fails to load under Unsloth, smoke-test first:
  uv run python train/unsloth_stage_a.py --smoke-load-only
and see docs/UNSLOTH.md for fallbacks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.prompts import POINT_PROMPT


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
        # Deep-ish copy of message structure with freshly opened image
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

    rows = load_jsonl(args.data)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    # Load model first (onto GPU) before touching the image corpus.
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
        f"samples={len(dataset)} from {args.data} "
        f"(lazy images, max_side={args.max_image_side or 'none'})"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.lr,
            logging_steps=10,
            save_steps=args.save_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=str(args.out),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=args.max_seq_length,
            bf16=True,
            dataloader_num_workers=args.num_workers,
            dataloader_pin_memory=False,
        ),
    )
    trainer.train()
    model.save_pretrained(str(args.out / "lora_adapter"))
    tokenizer.save_pretrained(str(args.out / "lora_adapter"))
    print(f"Saved LoRA → {args.out / 'lora_adapter'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unsloth Stage A POINT SFT")
    ap.add_argument("--data", type=Path, default=ROOT / "data/splits/train.jsonl")
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints/stage_a_point_unsloth")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 overrides epochs")
    ap.add_argument("--max-samples", type=int, default=0, help="debug subset; 0=all")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=100)
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

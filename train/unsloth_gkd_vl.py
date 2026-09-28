#!/usr/bin/env python3
"""Vision-language on-policy distillation (OPD) = TRL `GKDTrainer` + a thin VL shim.

Why this file exists
--------------------
`trl.GKDTrainer` already implements the OvisOCR2 paper's recipe
(``lmbda=1`` → pure on-policy; ``beta=1`` → reverse KL, i.e. mode-seeking), and it
only computes logits on the *completion* span, so full-vocab KL is cheap even at
12k-token prompts.  What it lacks is multimodal plumbing: ``compute_loss`` and
``generate_on_policy_outputs`` forward only ``input_ids``/``attention_mask``, so
``pixel_values`` never reach the student or the teacher.

This module adds exactly that:

  * ``VLGKDTrainer``   – subclass that passes multimodal kwargs through
  * ``VLDataCollator`` – {messages + PIL images} → prompts / input_ids / labels
                         (+ pixel_values / image_grid_thw / mm_token_type_ids)
  * ``--selfcheck``    – on-device plumbing validation (see below)

Selfcheck (the point of the exercise)
-------------------------------------
  (a) teacher == student          → reverse KL must be ~0 (only float noise)
  (b) random-LoRA student vs frozen base teacher → loss > 0 and grads flow
If (a) is not ~0 the prompt/label slicing or the multimodal wiring is wrong; if
(b) shows no gradients the student path is disconnected.

Usage
-----
  # 逻辑自检（本地即可，8GB 够用；用 448² 小分辨率加速）
  uv run python train/unsloth_gkd_vl.py --selfcheck --model models/Qwen3.5-0.8B

  # 真训练（teacher 是另一个模型；学生 = --model (+ --student-adapter)）
  uv run python train/unsloth_gkd_vl.py \
    --model models/Qwen3.5-0.8B --student-adapter checkpoints/q2_08b/adapter_final \
    --teacher checkpoints/q2_4b_merged \
    --data data/splits_grpo_q2_hq200/train.jsonl \
    --max-seq-length 12288 --max-pixels 8294400 --max-prompt-length 10240 \
    --lmbda 1.0 --beta 1.0 --temperature 1.0 --max-new-tokens 256 \
    --mask-translation --lr 1e-6 --epochs 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "train"))

MM_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
)


def patch_eos_placeholder(processor) -> None:
    """Map the literal ``'<EOS_TOKEN>'`` to the real eos id on this tokenizer.

    On this stack (Unsloth's config-compat layer + TRL 0.24) ``SFTTrainer`` can
    resolve ``args.eos_token`` to the placeholder string ``'<EOS_TOKEN>'``, which
    is not in the Qwen vocabulary, and then raises. Resolving it to the model's
    real eos id keeps the check happy and is a no-op otherwise.
    """
    tok = getattr(processor, "tokenizer", processor)
    if getattr(tok, "_point_eos_patched", False):
        return
    orig = tok.convert_tokens_to_ids

    def wrapper(token):
        val = orig(token)
        if val is None and str(token) == "<EOS_TOKEN>":
            return tok.eos_token_id
        return val

    tok.convert_tokens_to_ids = wrapper
    tok._point_eos_patched = True


def _mm(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in inputs.items() if k in MM_KEYS and v is not None}


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
def make_vl_gkd_trainer():
    import torch
    from trl import GKDTrainer

    class VLGKDTrainer(GKDTrainer):
        """GKDTrainer that forwards pixel_values/image_grid_thw to both models."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if getattr(self, "use_liger_gkd_loss", False):  # not enabled here
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            mm = _mm(inputs)
            student_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **mm,
            )
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **mm,
                )
            prompt_len = inputs["prompts"].shape[1]
            loss = self.generalized_jsd_loss(
                student_logits=student_outputs.logits[:, prompt_len - 1 : -1, :],
                teacher_logits=teacher_outputs.logits[:, prompt_len - 1 : -1, :],
                labels=inputs["labels"][:, prompt_len:],
                beta=self.beta,
            )
            return (loss, student_outputs) if return_outputs else loss

        @staticmethod
        def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs["prompts"],
                    attention_mask=inputs.get("prompt_attention_mask"),
                    **_mm(inputs),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )
            ids = out.sequences
            attn = torch.ones_like(ids)
            labels = ids.clone()
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
                attn[ids == pad_token_id] = 0
            return ids, attn, labels

    return VLGKDTrainer


# --------------------------------------------------------------------------- #
# collator
# --------------------------------------------------------------------------- #
class VLDataCollator:
    """Batch {messages, images} into TRL-GKD expected keys, with vision.

    Layout: ``[ prompt block (fixed --max-prompt-length, LEFT padded) |
                completion block (RIGHT padded) ]``.
    A fixed prompt width is required because GKDTrainer slices logits with a
    single ``prompts.shape[1]``.
    """

    def __init__(
        self,
        processor,
        *,
        max_prompt_length: int = 10240,
        max_length: int = 12288,
        mask_translation: bool = False,
    ) -> None:
        self.proc = processor
        inner = getattr(processor, "tokenizer", processor)
        self.inner = inner
        self.pad_id = getattr(inner, "pad_token_id", 0) or 0
        self.img_id = inner.convert_tokens_to_ids("<|image_pad|>")
        self.eos_src_id = inner.convert_tokens_to_ids("</source>")
        self.max_prompt_length = int(max_prompt_length)
        self.max_length = int(max_length)
        self.mask_translation = bool(mask_translation)
        self.last_stats: dict[str, Any] = {}

    # -- helpers ----------------------------------------------------------- #
    def _texts(self, ex: dict[str, Any]) -> tuple[str, str, list[Any]]:
        msgs = ex["messages"]
        images = [
            p["image"]
            for m in msgs
            for p in (m.get("content") or [])
            if isinstance(p, dict) and p.get("type") == "image"
        ]
        user_only = [m for m in msgs if m.get("role") == "user"]
        prompt = self.proc.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)
        full = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return prompt, full, images

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        prompts_txt, fulls_txt, images = [], [], []
        for ex in examples:
            p, f, imgs = self._texts(ex)
            prompts_txt.append(p)
            fulls_txt.append(f)
            images.extend(imgs)

        full = self.proc(text=fulls_txt, images=images or None, padding=True, return_tensors="pt")
        self.inner.padding_side = "left"
        pr = self.proc(text=prompts_txt, images=images or None, padding=True, return_tensors="pt")

        full_ids = full["input_ids"]
        full_attn = full["attention_mask"]
        pr_ids = pr["input_ids"]
        pr_attn = pr["attention_mask"]
        b = full_ids.shape[0]
        P = self.max_prompt_length

        prompt_block = torch.full((b, P), self.pad_id, dtype=full_ids.dtype)
        prompt_mask = torch.zeros((b, P), dtype=full_attn.dtype)
        comps: list[torch.Tensor] = []
        prompt_lens: list[int] = []
        for i in range(b):
            p_ids = pr_ids[i][pr_attn[i].bool()]
            lp = int(p_ids.numel())
            la = int(full_attn[i].sum()) - lp
            comp = full_ids[i][lp : lp + la]
            if lp > P:
                raise ValueError(
                    f"prompt {lp} tokens > --max-prompt-length {P}; raise it "
                    f"(vision tokens grow with --max-pixels)"
                )
            prompt_block[i, P - lp :] = p_ids
            prompt_mask[i, P - lp :] = 1
            comps.append(comp)
            prompt_lens.append(lp)

        C = max(1, max(int(c.numel()) for c in comps))
        if P + C > self.max_length:
            C = max(1, self.max_length - P)
        comp_block = torch.full((b, C), self.pad_id, dtype=full_ids.dtype)
        comp_mask = torch.zeros((b, C), dtype=full_attn.dtype)
        for i, c in enumerate(comps):
            c = c[:C]
            comp_block[i, : c.numel()] = c
            comp_mask[i, : c.numel()] = 1

        input_ids = torch.cat([prompt_block, comp_block], dim=1)
        attention_mask = torch.cat([prompt_mask, comp_mask], dim=1)
        labels = input_ids.clone()
        labels[:, :P] = -100
        labels[attention_mask == 0] = -100

        if self.mask_translation and self.eos_src_id is not None and self.eos_src_id >= 0:
            for i in range(b):
                pos = (comp_block[i] == self.eos_src_id).nonzero(as_tuple=True)[0]
                if pos.numel():
                    labels[i, P + int(pos[0]) + 1 :] = -100

        mm_ids = (input_ids == self.img_id).long() if isinstance(self.img_id, int) and self.img_id >= 0 else None
        batch: dict[str, Any] = {
            "prompts": prompt_block,
            "prompt_attention_mask": prompt_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if mm_ids is not None:
            batch["mm_token_type_ids"] = mm_ids
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if k in full:
                batch[k] = full[k]
        self.last_stats = {
            "batch": b,
            "prompt_len_max": max(prompt_lens),
            "completion_max": int((labels != -100).sum(dim=1).max()),
            "seq_len": int(input_ids.shape[1]),
            "pixels": tuple(full["pixel_values"].shape) if "pixel_values" in full else None,
        }
        return batch


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #
def load_vl(path: str, *, max_seq_length: int, lora: dict[str, Any] | None = None, load_in_4bit: bool = True):
    from unsloth import FastVisionModel

    model, proc = FastVisionModel.from_pretrained(
        path, load_in_4bit=load_in_4bit, use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
    )
    if lora is not None:
        model = FastVisionModel.get_peft_model(model, **lora)
    return model, proc


def default_lora(seed: int = 42) -> dict[str, Any]:
    return dict(
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", random_state=seed,
        use_rslora=False, loftq_config=None, target_modules="all-linear",
    )


def perturb_lora(model, *, scale: float = 0.02, seed: int = 0) -> int:
    """Fill LoRA `lora_B` with small noise so the student differs from the teacher.

    LoRA initialises B to zero, so a freshly attached adapter is numerically
    identical to the teacher and reverse KL would be ~0 for the wrong reason.
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    n = 0
    for name, p in model.named_parameters():
        if "lora_B" in name and p.requires_grad:
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * scale)
            n += 1
    return n


def load_rows(path: Path, n: int) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if n and len(rows) >= n:
            break
    return rows


def stage_rows(rows: list[dict[str, Any]], *, min_pixels: int, max_pixels: int):
    """Normalise ShareGPT rows to {messages:[{role, content:[parts]}]} with PIL images.

    The split JSONLs store ``content`` as a string starting with ``<image>`` and the
    image path in ``row["images"][0]``; the collator wants explicit parts.
    """
    from PIL import Image

    from point_ocr.image_resize import resize_for_ovis

    out = []
    for r in rows:
        imgs = r.get("images") or []
        pil = None
        if imgs:
            with Image.open(str(imgs[0])) as im:
                pil = resize_for_ovis(im.convert("RGB"), min_pixels=min_pixels, max_pixels=max_pixels)
        msgs = []
        for m in r["messages"]:
            c = m.get("content")
            parts: list[dict[str, Any]] = []
            if isinstance(c, str):
                text = c
                if text.startswith("<image>") and pil is not None:
                    parts.append({"type": "image", "image": pil})
                    text = text[len("<image>") :]
                parts.append({"type": "text", "text": text})
            else:
                for part in c or []:
                    if isinstance(part, dict) and part.get("type") == "image":
                        parts.append({"type": "image", "image": pil})
                    else:
                        parts.append(part)
            msgs.append({"role": m["role"], "content": parts})
        out.append({"messages": msgs})
    return out


class ListDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ex = self.rows[i]
        return {"messages": ex["messages"]} if "messages" in ex else ex


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #
def selfcheck(args) -> int:
    import torch
    from trl import GKDConfig

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    VLGKDTrainer = make_vl_gkd_trainer()

    rows = load_rows(Path(args.data), args.max_samples or 8)
    staged = stage_rows(rows, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    ds = ListDataset(staged)

    def run(tag: str, student, teacher, proc, expect_zero: bool) -> float:
        collator = VLDataCollator(
            proc, max_prompt_length=args.max_prompt_length, max_length=args.max_seq_length,
            mask_translation=args.mask_translation,
        )
        cfg = GKDConfig(
            output_dir=f"/tmp/gkd_selfcheck_{tag}",
            eos_token=getattr(getattr(proc, "tokenizer", proc), "eos_token", None),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            max_steps=args.steps,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            bf16=True,
            max_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            lmbda=args.lmbda,
            beta=args.beta,
            temperature=args.temperature,
            disable_dropout=True,
            use_liger_kernel=False,
        )
        # Unsloth's patched trainer injects a default EOS_TOKEN constant when
        # args.eos_token is None; pin it to the real tokenizer token instead.
        inner_tok = getattr(proc, "tokenizer", proc)
        cfg.eos_token = getattr(inner_tok, "eos_token", None)
        cfg.remove_unused_columns = False
        patch_eos_placeholder(proc)
        print(f"[gkd] eos_token={cfg.eos_token!r} lmbda={cfg.lmbda} beta={cfg.beta} "
              f"max_new_tokens={cfg.max_new_tokens}", flush=True)
        trainer = VLGKDTrainer(
            model=student, teacher_model=teacher, args=cfg,
            data_collator=collator, train_dataset=ds, processing_class=proc,
        )
        print(f"[selfcheck:{tag}] collator stats = {collator.last_stats}", flush=True)
        out = trainer.train()
        hist = trainer.state.log_history
        loss = float(out.training_loss)
        losses = [float(h["loss"]) for h in hist if "loss" in h]
        gns = [float(h["grad_norm"]) for h in hist if h.get("grad_norm") is not None]
        gn = max(gns) if gns else 0.0
        print(f"[selfcheck:{tag}] steps={args.steps} losses={losses} loss={loss:.6f} "
              f"grad_norms={gns}", flush=True)
        print(f"[selfcheck:{tag}] collator stats = {collator.last_stats}", flush=True)
        if expect_zero:
            ok = loss < 1e-3
            print(f"[selfcheck:{tag}] {'PASS' if ok else 'FAIL'}: teacher==student => reverse KL ~ 0 "
                  f"(loss={loss:.2e})", flush=True)
        else:
            ok = loss > 1e-3 and gn > 0.0
            print(f"[selfcheck:{tag}] {'PASS' if ok else 'FAIL'}: perturbed student => loss>0 & grads flow "
                  f"(loss={loss:.4f}, grad_norm={gn:.3f})", flush=True)
        return loss if ok else -1.0

    print("=== selfcheck (a) teacher == student : reverse KL must be ~0 ===", flush=True)
    student_a, proc = load_vl(args.model, max_seq_length=args.max_seq_length, lora=default_lora(1))
    a = run("self", student_a, student_a, proc, expect_zero=True)
    del student_a
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    print("=== selfcheck (b) random-LoRA student vs frozen base teacher ===", flush=True)
    teacher_b, proc_b = load_vl(args.model, max_seq_length=args.max_seq_length, lora=None)
    student_b, _ = load_vl(args.model, max_seq_length=args.max_seq_length, lora=default_lora(7))
    n_pert = perturb_lora(student_b, scale=0.02, seed=7)
    print(f"[selfcheck] perturbed {n_pert} lora_B tensors so student != teacher", flush=True)
    b = run("perturbed", student_b, teacher_b, proc_b, expect_zero=False)

    ok = a >= 0.0 and b >= 0.0
    print(f"\n[selfcheck] {'ALL PASS' if ok else 'FAILED'}  (a={a:.6f}, b={b:.6f})", flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(args) -> int:
    from trl import GKDConfig

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    VLGKDTrainer = make_vl_gkd_trainer()

    lora = default_lora(args.seed)
    if args.student_adapter:
        # continue from a SFT adapter is not supported by get_peft_model; load base+adapter instead
        raise SystemExit("--student-adapter 需要 delta-merge 到 --model（用 export/merge_unsloth_lora.py 先烘焙）")
    student, proc = load_vl(args.model, max_seq_length=args.max_seq_length, lora=lora)
    if args.teacher.strip().lower() == "self":
        teacher = student
    else:
        teacher, _ = load_vl(args.teacher, max_seq_length=args.max_seq_length, lora=None)

    rows = load_rows(Path(args.data), args.max_samples)
    staged = stage_rows(rows, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    collator = VLDataCollator(
        proc, max_prompt_length=args.max_prompt_length, max_length=args.max_seq_length,
        mask_translation=args.mask_translation,
    )
    cfg = GKDConfig(
        output_dir=args.out,
        eos_token=getattr(getattr(proc, "tokenizer", proc), "eos_token", None),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to=["tensorboard"],
        bf16=True,
        max_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        disable_dropout=True,
        use_liger_kernel=False,
        seed=args.seed,
    )
    inner_tok = getattr(proc, "tokenizer", proc)
    cfg.eos_token = getattr(inner_tok, "eos_token", None)
    cfg.remove_unused_columns = False
    patch_eos_placeholder(proc)
    print(f"[gkd] eos_token={cfg.eos_token!r} lmbda={cfg.lmbda} beta={cfg.beta} "
          f"max_new_tokens={cfg.max_new_tokens}", flush=True)
    trainer = VLGKDTrainer(
        model=student, teacher_model=teacher, args=cfg,
        data_collator=collator, train_dataset=ListDataset(staged), processing_class=proc,
    )
    print(f"[gkd] rows={len(staged)} collator={collator.last_stats}", flush=True)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[gkd] saved → {args.out}", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true", help="run the (a)/(b) plumbing validation and exit")
    ap.add_argument("--model", default="models/Qwen3.5-0.8B", help="student base (or its merged SFT)")
    ap.add_argument("--student-adapter", default="", help="unsupported placeholder (bake first)")
    ap.add_argument("--teacher", default="self", help="teacher dir, or 'self'")
    ap.add_argument("--data", default="data/splits_grpo_q1_hq200/train.jsonl")
    ap.add_argument("--out", default="checkpoints/gkd_opd")
    ap.add_argument("--max-seq-length", type=int, default=12288)
    ap.add_argument("--max-prompt-length", type=int, default=10240)
    ap.add_argument("--max-pixels", type=int, default=8294400)
    ap.add_argument("--min-pixels", type=int, default=200704)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lmbda", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--mask-translation", action="store_true",
                    help="mask everything after </source> in the student labels (Q2: localisation-only loss)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2, help="selfcheck steps")
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("UNSLOTH_CE_LOSS_N_CHUNKS", "32")
    raise SystemExit(selfcheck(args) if args.selfcheck else train(args))


if __name__ == "__main__":
    main()

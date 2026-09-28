#!/usr/bin/env python3
"""First GRPO pass on q1_withreal hard scenes (multi_frag / short group / empty).

Bakes q1_withreal into the base as frozen π_ref, then trains a new LoRA.
Linux + NVIDIA only.

  uv run python train/unsloth_grpo.py \\
    --adapter checkpoints/q1_withreal/adapter_final \\
    --data data/splits_grpo_q1_hard/train.jsonl \\
    --model models/Qwen3.5-0.8B \\
    --run-id grpo_q1_hard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from math import lcm
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from transformers import TrainerCallback

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

from point_ocr.grpo_data import assistant_text, user_text  # noqa: E402
from point_ocr.grpo_rewards import (  # noqa: E402
    REWARD_FUNCS as Q1_REWARD_FUNCS,
    REWARD_WEIGHTS as Q1_REWARD_WEIGHTS,
)
from point_ocr.image_resize import resize_for_ovis, smart_resize_hw  # noqa: E402
from run_observability import (  # noqa: E402
    git_rev,
    package_versions,
    plot_grpo_metric_curves,
    plot_loss_curves,
    resolve_run_dir,
    run_final_generate_eval,
    write_loss_history,
    write_run_config,
)
from unsloth_stage_a import (  # noqa: E402
    _align_special_tokens,
    _cast_trainable_lora_,
    _ensure_vision_image_size,
    _preferred_compute_dtype,
    _quiet_unsloth_noise,
    _summarize_lora_dtypes,
    load_jsonl,
)


def _patch_trl_mergekit() -> None:
    """trl 0.24 `is_mergekit_available()` can return (False, None), which is truthy."""
    import trl.import_utils as iu

    iu._mergekit_available = False
    iu.is_mergekit_available = lambda: False  # type: ignore[method-assign]


def _cat_fw_tensors(vals: list):
    import torch

    x0 = vals[0]
    if not torch.is_tensor(x0):
        return vals[0]
    if all(tuple(v.shape[1:]) == tuple(x0.shape[1:]) for v in vals):
        return torch.cat(vals, dim=0)
    # token_type_ids etc: pad sequence dim 1, then cat batch.
    max1 = max(v.size(1) for v in vals)
    padded = []
    for v in vals:
        if v.size(1) < max1:
            v = torch.nn.functional.pad(v, (0, max1 - v.size(1)))
        padded.append(v)
    return torch.cat(padded, dim=0)


def _merge_forward_kwargs(parts: list[dict]) -> dict:
    if not parts:
        return {}
    keys: set[str] = set()
    for p in parts:
        keys |= set(p)
    out: dict = {}
    for k in keys:
        vs = [p[k] for p in parts if k in p and p[k] is not None]
        if not vs:
            continue
        out[k] = _cat_fw_tensors(vs)
    return out


def _chunked_grpo_trainer_cls(GRPOTrainer, *, gen_chunk_size: int):
    """G samples still; generate / logprob run in slices of ``gen_chunk_size``."""

    class ChunkedGRPOTrainer(GRPOTrainer):
        def _generate_single_turn(self, prompts, images):
            n = len(prompts)
            chunk = int(gen_chunk_size)
            if chunk <= 0 or n <= chunk:
                return super()._generate_single_turn(prompts, images)
            prompt_ids: list = []
            completion_ids: list = []
            logprobs = None
            fw_parts: list[dict] = []
            import torch

            for i in range(0, n, chunk):
                sl = slice(i, min(i + chunk, n))
                sub_img = images[sl] if images is not None else None
                p, c, lp, fw = super()._generate_single_turn(prompts[sl], sub_img)
                prompt_ids.extend(p)
                completion_ids.extend(c)
                if lp is not None:
                    logprobs = ([] if logprobs is None else logprobs)
                    logprobs.extend(lp)
                if fw:
                    fw_parts.append(fw)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return prompt_ids, completion_ids, logprobs, _merge_forward_kwargs(fw_parts)

        def _get_per_token_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=None,
            compute_entropy=False,
            compute_efficient=False,
            *args,
            **kwargs,
        ):
            # UnslothGRPOTrainer calls this with 7 positionals (incl. compute_entropy /
            # compute_efficient). **kwargs cannot absorb extra positionals.
            chunk = int(gen_chunk_size)
            if chunk > 0:
                batch_size = chunk if batch_size is None else min(int(batch_size), chunk)
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                compute_entropy,
                compute_efficient,
                *args,
                **kwargs,
            )

    ChunkedGRPOTrainer.__name__ = "ChunkedGRPOTrainer"
    return ChunkedGRPOTrainer


class GrpoLazyDataset:
    """Decode JPEGs in __getitem__; keep GRPO columns TRL needs for rewards."""

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
        self.rows = rows
        self.max_side = max_side
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        images = row.get("images") or []
        if not images:
            raise IndexError(f"row {idx} has no images")
        img = self._Image.open(str(images[0])).convert("RGB")
        if self.max_pixels and self.max_pixels > 0:
            img = resize_for_ovis(
                img,
                min_pixels=self.min_pixels or (448 * 448),
                max_pixels=self.max_pixels,
            )
        elif self.max_side and max(img.size) > self.max_side:
            img.thumbnail((self.max_side, self.max_side))
        prompt_text = user_text(row).strip()
        meta = row.get("metadata") or {}
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "image": img,
            "target": assistant_text(row),
            "is_negative": bool(meta.get("is_negative")),
            "pool_id": str(meta.get("pool_id") or ""),
            "grpo_scene": str(meta.get("grpo_scene") or ""),
        }


def _select_rewards(name: str) -> tuple[list, list[float]]:
    """q1 = legacy edit/empty/leak. q1v2 = +boundary penalties. q2 = OCR-MT."""
    key = str(name).lower()
    if key == "q2":
        from point_ocr.grpo_rewards_q2 import (
            REWARD_FUNCS as Q2_FUNCS,
            WEIGHT_FIELDS,
            weights_from_env,
        )

        w = weights_from_env()  # honours GRPO_REWARD_WEIGHTS (localization-first etc.)
        return list(Q2_FUNCS), [float(getattr(w, f)) for f in WEIGHT_FIELDS]
    if key == "q1v2":
        from point_ocr.grpo_rewards_q1 import REWARD_FUNCS as F, REWARD_WEIGHTS as W

        return list(F), list(W)
    return list(Q1_REWARD_FUNCS), list(Q1_REWARD_WEIGHTS)


def _vision_tokens(h: int, w: int, *, min_pixels: int, max_pixels: int) -> int:
    """Qwen3.5: patch 16, spatial merge 2 → 32px per visual token."""
    nh, nw = smart_resize_hw(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
    return (nh // 32) * (nw // 32)


def _log_image_budget(rows: list[dict], *, g: int, min_pixels: int, max_pixels: int) -> None:
    from PIL import Image

    toks: list[int] = []
    for row in rows:
        path = (row.get("images") or [None])[0]
        if not path:
            continue
        with Image.open(str(path)) as im:
            w, h = im.size
        toks.append(_vision_tokens(h, w, min_pixels=min_pixels, max_pixels=max_pixels))
    if not toks:
        return
    toks.sort()
    p50 = toks[len(toks) // 2]
    p90 = toks[int(0.9 * (len(toks) - 1))]
    peak = toks[-1]
    print(
        f"GRPO image budget G={g} max_pixels={max_pixels}: "
        f"vis_tokens p50={p50} p90={p90} max={peak} → "
        f"generate batch ≈ {g}×{peak} = {g * peak} tokens",
        flush=True,
    )
    if g * peak > 16_000:
        print(
            "WARNING: G×max vis tokens is large; 24GB cards OOM on ~8k-token pages "
            "because TRL duplicates the image G times for generate + policy + π_ref. "
            "Lower --gen-chunk-size or --num-generations.",
            flush=True,
        )


def _freeze_vision_lora(model) -> int:
    n_freeze = 0
    for name, param in model.named_parameters():
        low = name.lower()
        if "lora" not in low:
            continue
        if any(k in low for k in ("vision", "visual", "image", "patch")):
            if param.requires_grad:
                param.requires_grad = False
                n_freeze += 1
    return n_freeze


def _bake_sft_as_ref(model, adapter: Path):
    """Merge SFT LoRA into the base so TRL disable_adapter() == frozen q1_withreal.

    GRPO then attaches a *new* LoRA. At step 0, π_θ = π_ref = merged SFT.
    Without this bake, TRL PEFT uses disable_adapter() → Instruct, not SFT.
    """
    from peft import PeftModel

    adapter = adapter.resolve()
    cfg_path = adapter / "adapter_config.json"
    if not adapter.is_dir() or not cfg_path.is_file():
        raise SystemExit(f"--adapter is not a PEFT dir: {adapter}")
    peft_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print(
        f"bake SFT → frozen π_ref from {adapter}\n"
        f"  peft_type={peft_cfg.get('peft_type')} r={peft_cfg.get('r')} "
        f"lora_alpha={peft_cfg.get('lora_alpha')}",
        flush=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    if not hasattr(model, "merge_and_unload"):
        raise SystemExit("loaded adapter cannot merge_and_unload; cannot pin π_ref to SFT")
    model = model.merge_and_unload()
    print("merged SFT into base; TRL KL disable_adapter() will be q1_withreal, not Instruct", flush=True)
    return model, peft_cfg


def _attach_grpo_lora(model, args: argparse.Namespace):
    from unsloth import FastVisionModel

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
    if not args.finetune_vision:
        n_freeze = _freeze_vision_lora(model)
        print(f"froze {n_freeze} vision LoRA tensors on the GRPO adapter", flush=True)
    return model


MERGED_SFT_CACHE = ROOT / "checkpoints" / "q1_withreal_merged"


def _merged_cache_for_adapter(adapter: Path) -> Path:
    """π_ref bake cache keyed by the adapter being merged (not always SFT)."""
    name = adapter.resolve().parent.name
    return ROOT / "checkpoints" / f"{name}_merged"


def _adapter_base_model(adapter: Path) -> str:
    cfg_path = adapter / "adapter_config.json"
    if not cfg_path.is_file():
        return ""
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(cfg.get("base_model_name_or_path") or "").strip()


def _resolve_peft_base(adapter: Path, fallback_model: str) -> str:
    """Local PEFT base_model_name_or_path wins over --model when present."""
    expected = _adapter_base_model(adapter)
    if expected:
        exp_path = Path(expected)
        if exp_path.is_dir():
            return str(exp_path.resolve())
        alt = ROOT / expected
        if alt.is_dir():
            return str(alt.resolve())
    return fallback_model


def _load_continue_adapter(args: argparse.Namespace):
    """Load PEFT base + existing LoRA as trainable (no bake, no new adapter).

    TRL ``disable_adapter()`` == the PEFT base (for GRPO_1: merged SFT), so
    KL(β) regularizes toward SFT while π_θ keeps updating the same LoRA.
    """
    from peft import PeftModel

    if not args.adapter:
        raise SystemExit("--continue-adapter requires --adapter")
    adapter = args.adapter.resolve()
    cfg_path = adapter / "adapter_config.json"
    if not adapter.is_dir() or not cfg_path.is_file():
        raise SystemExit(f"--adapter is not a PEFT dir: {adapter}")
    peft_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_id = _resolve_peft_base(adapter, args.model)
    use_4bit = not args.no_4bit
    print(
        f"continue-adapter: base={base_id}\n"
        f"  load trainable LoRA from {adapter}\n"
        f"  peft_type={peft_cfg.get('peft_type')} r={peft_cfg.get('r')} "
        f"lora_alpha={peft_cfg.get('lora_alpha')}\n"
        f"  KL disable_adapter() → PEFT base (not a baked copy of this LoRA)\n"
        f"  next: load base weights (may take ~15–60s; wait for 'loaded …' line)",
        flush=True,
    )
    model, tokenizer = _unsloth_from_pretrained(
        base_id, load_in_4bit=use_4bit, max_seq_length=args.max_seq_length
    )
    print(f"attaching trainable LoRA {adapter} …", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    print("continue-adapter ready", flush=True)
    if not args.finetune_vision:
        n_freeze = _freeze_vision_lora(model)
        print(f"froze {n_freeze} vision LoRA tensors on continued adapter", flush=True)
    return model, tokenizer, peft_cfg


def _adapter_stamp(adapter: Path) -> str:
    weight = adapter / "adapter_model.safetensors"
    if not weight.is_file():
        weight = adapter / "adapter_model.bin"
    st = weight.stat() if weight.is_file() else None
    return f"{adapter.resolve()}|{getattr(st, 'st_mtime_ns', 0)}|{getattr(st, 'st_size', 0)}"


def _merged_cache_fresh(cache: Path, adapter: Path) -> bool:
    stamp = cache / ".baked_from"
    return (
        (cache / "config.json").is_file()
        and stamp.is_file()
        and stamp.read_text(encoding="utf-8") == _adapter_stamp(adapter)
    )


def _left_pad_tokenizer(tokenizer) -> None:
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    inner.padding_side = "left"
    if getattr(tokenizer, "padding_side", None) is not None:
        tokenizer.padding_side = "left"


def _unsloth_from_pretrained(model_id: str, *, load_in_4bit: bool, max_seq_length: int):
    import os
    import time

    from unsloth import FastVisionModel

    local = Path(model_id)
    is_local = local.is_dir() or (ROOT / model_id).is_dir()
    if is_local and not local.is_dir():
        local = ROOT / model_id
        model_id = str(local.resolve())
    # Unsloth still pings the Hub after its banner even for local dirs; that
    # looks like a hang (often 30–120s with no progress). Force offline when
    # the checkpoint is already on disk.
    prev_hub = os.environ.get("HF_HUB_OFFLINE")
    prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    if is_local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    kw: dict = dict(
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=_gc_mode(),
        max_seq_length=max_seq_length,
    )
    if not load_in_4bit:
        kw["dtype"] = _preferred_compute_dtype()
    if is_local:
        kw["local_files_only"] = True
    print(
        f"loading {'4bit' if load_in_4bit else 'bf16'} from {model_id}"
        f"{' (local_files_only)' if is_local else ''} …",
        flush=True,
    )
    t0 = time.time()
    try:
        model, tokenizer = FastVisionModel.from_pretrained(model_id, **kw)
    finally:
        if is_local:
            if prev_hub is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_hub
            if prev_tf is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = prev_tf
    _align_special_tokens(model, tokenizer)
    _ensure_vision_image_size(model)
    _left_pad_tokenizer(tokenizer)
    print(
        f"loaded {model_id} load_in_4bit={load_in_4bit} in {time.time() - t0:.1f}s",
        flush=True,
    )
    return model, tokenizer


def _save_merged_cache(model, tokenizer, cache: Path, adapter: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    print(f"saving merged SFT π_ref → {cache}", flush=True)
    model.save_pretrained(str(cache))
    tokenizer.save_pretrained(str(cache))
    (cache / ".baked_from").write_text(_adapter_stamp(adapter), encoding="utf-8")


def _load_policy(args: argparse.Namespace):
    """Load π_ref / trainable policy.

    Default: merge --adapter into its base as frozen π_ref, then caller attaches
    a *new* LoRA. With --continue-adapter: load base + that LoRA as trainable.
    """
    import gc

    import torch

    if getattr(args, "continue_adapter", False):
        return _load_continue_adapter(args)

    adapter = args.adapter.resolve() if args.adapter else None
    use_4bit = not args.no_4bit
    peft_cfg = None
    cache = _merged_cache_for_adapter(adapter) if adapter else MERGED_SFT_CACHE

    if adapter and use_4bit and _merged_cache_fresh(cache, adapter):
        print(f"4bit π_ref cache hit: {cache}", flush=True)
        model, tokenizer = _unsloth_from_pretrained(
            str(cache), load_in_4bit=True, max_seq_length=args.max_seq_length
        )
        peft_cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        return model, tokenizer, peft_cfg

    if adapter is None:
        model, tokenizer = _unsloth_from_pretrained(
            args.model, load_in_4bit=use_4bit, max_seq_length=args.max_seq_length
        )
        return model, tokenizer, None

    base_id = _resolve_peft_base(adapter, args.model)
    print(f"π_ref bake base={base_id} adapter={adapter}", flush=True)

    # Merge in bf16 first (in-place 4bit merge is unreliable).
    model, tokenizer = _unsloth_from_pretrained(
        base_id, load_in_4bit=False, max_seq_length=args.max_seq_length
    )
    model, peft_cfg = _bake_sft_as_ref(model, adapter)
    if use_4bit:
        _save_merged_cache(model, tokenizer, cache, adapter)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer = _unsloth_from_pretrained(
            str(cache), load_in_4bit=True, max_seq_length=args.max_seq_length
        )
    return model, tokenizer, peft_cfg


class GreedyValCallback(TrainerCallback):
    """Greedy POINT eval on val jsonl at each checkpoint (save_steps)."""

    def __init__(
        self,
        *,
        jsonl_path: Path,
        run_dir: Path,
        max_pixels: int,
        min_pixels: int,
        max_image_side: int,
        hit_threshold: float,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.run_dir = Path(run_dir)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_image_side = max_image_side
        self.hit_threshold = hit_threshold

    def on_save(self, args, state, control, model=None, processing_class=None, **kwargs):
        del args, control, kwargs
        if model is None or not self.jsonl_path.is_file():
            return
        from unsloth import FastVisionModel

        step_dir = self.run_dir / "val_steps" / f"step_{int(state.global_step)}"
        step_dir.mkdir(parents=True, exist_ok=True)
        FastVisionModel.for_inference(model)
        try:
            report = run_final_generate_eval(
                model=model,
                tokenizer=processing_class,
                jsonl_path=self.jsonl_path,
                run_dir=step_dir,
                max_samples=0,
                hit_threshold=self.hit_threshold,
                max_image_side=self.max_image_side,
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels,
            )
            overall = report.get("overall") or {}
            payload = {
                "overall": overall,
                "by_grpo_scene": report.get("by_grpo_scene") or {},
            }
            out = self.run_dir / "metrics" / f"val_step_{int(state.global_step)}.json"
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            scenes = payload["by_grpo_scene"]
            scene_bits = " ".join(
                f"{k}={v.get('block_hit_rate')}"
                for k, v in sorted(scenes.items())
                if isinstance(v, dict)
            )
            print(
                f"greedy val step={int(state.global_step)} "
                f"hit={overall.get('block_hit_rate')} "
                f"{scene_bits} → {out}",
                flush=True,
            )
        except Exception as exc:
            print(f"[warn] greedy val failed at step {state.global_step}: {exc}", flush=True)
        finally:
            FastVisionModel.for_training(model)


class GrpoCurvesCallback(TrainerCallback):
    """Rewrite metrics/curves/*.png from trainer log_history (one PNG per metric)."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def _write(self, state) -> None:
        history = list(getattr(state, "log_history", None) or [])
        if not history:
            return
        write_loss_history(self.run_dir, history)
        plot_grpo_metric_curves(self.run_dir, history)

    def on_log(self, args, state, control, **kwargs):
        del args, control, kwargs
        self._write(state)

    def on_train_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        self._write(state)


def train(args: argparse.Namespace) -> None:
    _quiet_unsloth_noise()
    os.environ.setdefault("UNSLOTH_DATASET_NUM_PROC", "2")
    if getattr(args, "tf32", False):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    import unsloth  # noqa: F401
    from unsloth import FastVisionModel

    _quiet_unsloth_noise()
    _patch_trl_mergekit()
    from trl import GRPOConfig, GRPOTrainer

    started_at = datetime.now().astimezone()
    run_dir = resolve_run_dir(
        out=args.out,
        out_root=args.out_root,
        run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_grpo",
        started_at=started_at,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir.resolve()
    tb_dir = (run_dir / "tb").resolve()
    tb_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)

    rows = load_jsonl(args.data)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]
    dataset = GrpoLazyDataset(
        rows,
        max_side=args.max_image_side,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No usable samples from {args.data}")

    reward_funcs, reward_weights = _select_rewards(getattr(args, "reward_set", "q1"))
    run_config: dict = {
        "started_at": started_at.isoformat(),
        "run_dir": str(run_dir),
        "git_commit": git_rev(),
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_train_rows_loaded": len(rows),
        "packages": package_versions(),
        "tensorboard_logdir": str(tb_dir),
        "reward_set": getattr(args, "reward_set", "q1"),
        "reward_funcs": [fn.__name__ for fn in reward_funcs],
        "reward_weights": list(reward_weights),
    }
    write_run_config(run_dir, run_config)
    print(f"run_dir = {run_dir}")
    print(f"train prompts={len(dataset)} G={args.num_generations} from {args.data}")
    _log_image_budget(
        rows,
        g=int(args.num_generations),
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    model, tokenizer, peft_cfg = _load_policy(args)
    if args.continue_adapter:
        base_id = _resolve_peft_base(args.adapter.resolve(), args.model)
        run_config["pi_ref"] = {
            "kind": "continue_adapter",
            "path": str(args.adapter.resolve()),
            "peft_base": base_id,
            "r": (peft_cfg or {}).get("r"),
            "lora_alpha": (peft_cfg or {}).get("lora_alpha"),
            "load_in_4bit": not args.no_4bit,
            "note": "Trainable LoRA continued from --adapter. "
            "TRL disable_adapter() == PEFT base (SFT merged for GRPO_1).",
        }
        write_run_config(run_dir, run_config)
        # Keep CLI rank/alpha in sync with the loaded adapter for logging.
        if peft_cfg:
            if peft_cfg.get("r") is not None:
                args.lora_rank = int(peft_cfg["r"])
            if peft_cfg.get("lora_alpha") is not None:
                args.lora_alpha = int(peft_cfg["lora_alpha"])
    elif args.adapter:
        cache = _merged_cache_for_adapter(args.adapter.resolve())
        run_config["pi_ref"] = {
            "kind": "merged_adapter_qlora" if not args.no_4bit else "merged_adapter",
            "path": str(args.adapter.resolve()),
            "sft_r": (peft_cfg or {}).get("r"),
            "sft_lora_alpha": (peft_cfg or {}).get("lora_alpha"),
            "load_in_4bit": not args.no_4bit,
            "merged_cache": str(cache),
            "note": "Adapter merged into its PEFT base as frozen π_ref; GRPO trains a new LoRA. "
            "TRL disable_adapter() == baked π_ref.",
        }
        write_run_config(run_dir, run_config)
    else:
        run_config["pi_ref"] = {
            "kind": "instruct_base",
            "load_in_4bit": not args.no_4bit,
            "note": "No --adapter; KL (if beta>0) is against the Instruct checkpoint.",
        }
        write_run_config(run_dir, run_config)
    if not args.continue_adapter:
        model = _attach_grpo_lora(model, args)
    FastVisionModel.for_training(model)
    _align_special_tokens(model, tokenizer)

    if args.no_4bit:
        target = _preferred_compute_dtype()
        n_cast = _cast_trainable_lora_(model, target)
        if n_cast:
            print(f"cast {n_cast} trainable LoRA tensors → {target}", flush=True)
    n_lora, lora_dtypes = _summarize_lora_dtypes(model)
    print(
        f"QLoRA={not args.no_4bit} r={args.lora_rank} α={args.lora_alpha} "
        f"trainable LoRA tensors={n_lora} dtypes (numel): {lora_dtypes}",
        flush=True,
    )

    warmup_steps = float(args.warmup_ratio)
    gen_bs = lcm(int(args.num_generations), int(args.batch_size))
    # Unsloth names this "mini batch" but uses it as the number of chunks:
    # samples_per_forward = ceil(N / unsloth_grpo_mini_batch). Autotune only
    # budgets text hidden/logits, so it one-shots all G vision forwards with
    # grads and OOMs on 2880². Force 1 image per loss / ref-logprob forward.
    n_loss_chunks = int(gen_bs)
    print(
        f"GRPO G={args.num_generations} beta={args.beta} "
        f"batch={args.batch_size} accum={args.grad_accum} generation_batch_size={gen_bs} "
        f"gen_chunk_size={args.gen_chunk_size} unsloth_grpo_mini_batch={n_loss_chunks} "
        f"(1 image per loss forward)",
        flush=True,
    )
    grpo_kwargs: dict = dict(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=["tensorboard"],
        seed=args.seed,
        bf16=True,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        generation_batch_size=gen_bs,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length if args.max_prompt_length > 0 else None,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        reward_weights=list(reward_weights),
        disable_dropout=True,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=bool(getattr(args, "pin_memory", True) and args.num_workers > 0),
        dataloader_persistent_workers=bool(getattr(args, "persistent_workers", True) and args.num_workers > 0),
        dataloader_prefetch_factor=(int(getattr(args, "prefetch_factor", 4)) if args.num_workers > 0 else None),
        eval_strategy="no",
        unsloth_grpo_mini_batch=n_loss_chunks,
    )
    try:
        grpo_args = GRPOConfig(**grpo_kwargs)
    except TypeError:
        grpo_kwargs.pop("unsloth_grpo_mini_batch", None)
        grpo_args = GRPOConfig(**grpo_kwargs)
    grpo_args.unsloth_grpo_mini_batch = n_loss_chunks
    grpo_args.report_to = ["tensorboard"]

    TrainerCls = GRPOTrainer
    if int(args.gen_chunk_size) > 0:
        TrainerCls = _chunked_grpo_trainer_cls(GRPOTrainer, gen_chunk_size=int(args.gen_chunk_size))
    trainer = TrainerCls(
        model=model,
        processing_class=tokenizer,
        reward_funcs=list(reward_funcs),
        args=grpo_args,
        train_dataset=dataset,
    )
    val_path = Path(args.val) if args.val else None
    if val_path is not None and val_path.is_file() and not args.skip_post_eval:
        trainer.add_callback(
            GreedyValCallback(
                jsonl_path=val_path,
                run_dir=run_dir,
                max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
                max_image_side=args.max_image_side,
                hit_threshold=args.hit_threshold,
            )
        )
        print(f"greedy val callback on {val_path} every save_steps={args.save_steps}", flush=True)
    trainer.add_callback(GrpoCurvesCallback(run_dir))
    logging.getLogger("transformers").setLevel(logging.INFO)
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
    grpo_pngs = plot_grpo_metric_curves(run_dir, log_history)
    if grpo_pngs:
        print(f"GRPO curves ({len(grpo_pngs)}) → {run_dir / 'metrics' / 'curves'}")

    ended_at = datetime.now().astimezone()
    run_config.update(
        {
            "ended_at": ended_at.isoformat(),
            "duration_sec": (ended_at - started_at).total_seconds(),
            "train_result": {
                k: (float(v) if hasattr(v, "item") else v)
                for k, v in dict(train_result.metrics).items()
            },
            "adapter_final": str(adapter_dir),
            "n_train_dataset": len(dataset),
        }
    )
    run_config["packages"] = package_versions()
    write_run_config(run_dir, run_config)

    if not args.skip_post_eval:
        post_path = args.post_eval or args.val
        if post_path is None or not Path(post_path).exists():
            print("[warn] skip post-eval: no --post-eval / --val path", file=sys.stderr)
        else:
            print(f"post-train generate eval on {post_path} ...")
            FastVisionModel.for_inference(model)
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
            )
            run_config["post_eval"] = {
                "path": str(post_path),
                "overall": report.get("overall"),
            }
            write_run_config(run_dir, run_config)

    print(f"done. run_dir={run_dir}")
    print(f"AutoDL TensorBoard → {tb_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unsloth POINT GRPO (continues from a baked π_ref adapter)")
    ap.add_argument("--data", type=Path, default=ROOT / "data/splits_grpo_q2_marker/train.jsonl")
    ap.add_argument("--val", type=Path, default=ROOT / "data/splits_grpo_q2_marker/val.jsonl")
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "checkpoints/grpo_q1_signal/adapter_final",
        help="With default bake mode: merged as frozen π_ref, then a new LoRA is trained. "
        "With --continue-adapter: this LoRA is loaded trainable (no new adapter). "
        "PEFT base is read from adapter_config when local.",
    )
    ap.add_argument(
        "--continue-adapter",
        action="store_true",
        help="Continue training the existing --adapter LoRA on its PEFT base "
        "(do not bake / do not attach a new LoRA). KL disable_adapter() == PEFT base.",
    )
    ap.add_argument("--out-root", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--run-id", default="grpo_q2_marker")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Completions per device per step. Default matches --num-generations (8).",
    )
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lr-scheduler", default="cosine")
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=2880 * 2880,
        help="Smart-resize area cap (same as SFT). Generate is chunked by --gen-chunk-size, not full G.",
    )
    ap.add_argument("--min-pixels", type=int, default=448 * 448)
    ap.add_argument("--max-image-side", type=int, default=0)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Keep 0 for GRPO: TRL duplicates each image G times and worker shared "
        "memory can exhaust host RAM (observed host OOM at 4 workers).",
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
    )
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 matmul/cudnn on Ampere+.",
    )
    ap.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load merged π_ref in bf16 instead of 4bit QLoRA (more VRAM).",
    )
    ap.add_argument(
        "--finetune-vision",
        action="store_true",
        help="Keep vision LoRA trainable (default: freeze vision LoRA).",
    )
    ap.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume an interrupted run from a Trainer checkpoint dir.",
    )
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument(
        "--gen-chunk-size",
        type=int,
        default=2,
        help="How many of the G completions to generate (and score) at once. "
        "G itself is unchanged. 0 = TRL default (all G in one generate).",
    )
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument(
        "--reward-set",
        choices=["q1", "q1v2", "q2"],
        default="q1",
        help="q1 = edit/empty/leak. q1v2 = +over/under-extraction (Q1 v2). "
        "q2 = source/chrF/xml/empty/hallucination/over/leak (OCR-MT XML targets).",
    )
    ap.add_argument(
        "--max-prompt-length",
        type=int,
        default=0,
        help="0 disables truncation (required for vision chat templates).",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL weight to frozen merged SFT (q1_withreal). 0 disables.",
    )
    ap.add_argument("--post-eval", type=Path, default=None)
    ap.add_argument("--post-eval-max", type=int, default=0)
    ap.add_argument("--hit-threshold", type=float, default=0.85)
    ap.add_argument("--skip-post-eval", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()

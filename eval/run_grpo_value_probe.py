#!/usr/bin/env python3
"""G=8 rollout probe: which GRPO-eligible SFT rows have a learning signal.

Default pool: multi_frag + semantic_group from q1_withreal train (~5600).
Frozen q1_withreal, same weighted reward as GRPO. GPU generate batches G
copies of each image (falls back on OOM).

  uv run python eval/run_grpo_value_probe.py
  bash eval/run_grpo_value_probe.sh

Writes checkpoints/grpo_value_probe/{rows.jsonl,summary.json,run_config.json}.
Resume-safe: already-scored sample_ids are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "train"))

import unsloth  # noqa: E402,F401

from point_ocr.dataset_format import load_jsonl  # noqa: E402
from point_ocr.grpo_data import assistant_text, grpo_scene, user_text, val_slice  # noqa: E402
from point_ocr.pools.spec import quota_from_frac  # noqa: E402
from point_ocr.grpo_rewards import (  # noqa: E402
    cleaned_pred,
    group_learning_verdict,
    group_reward_stats,
    weighted_rewards,
)
from point_ocr.image_resize import resize_for_ovis, smart_resize_hw  # noqa: E402
from point_ocr.infer import apply_point_chat_template, point_eos_token_ids  # noqa: E402
from unsloth_grpo import MERGED_SFT_CACHE, _load_policy  # noqa: E402
from unsloth_stage_a import _quiet_unsloth_noise  # noqa: E402

DEFAULT_SCENES = ("multi_frag", "semantic_group")


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") or row.get("meta") or {}


def _vision_tokens(h: int, w: int, *, min_pixels: int, max_pixels: int) -> int:
    nh, nw = smart_resize_hw(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
    return (nh // 32) * (nw // 32)


def _reward_fn(reward_set: str):
    key = str(reward_set or "q1").lower()
    if key == "q1v2":
        from point_ocr.grpo_rewards_q1 import weighted_rewards as fn

        return fn
    if key == "q2":
        from point_ocr.grpo_rewards_q2 import weighted_rewards_q2 as fn

        return fn
    from point_ocr.grpo_rewards import weighted_rewards as fn

    return fn


def select_probe_rows(
    rows: list[dict[str, Any]],
    *,
    scenes: set[str],
    max_samples: int,
    seed: int = 42,
    scene_mix: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Pick GRPO-eligible rows.

    With ``max_samples`` the cap is spread across scenes: by ``scene_mix`` when
    given (match the compose recipe), else proportionally to availability.
    """
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scene = grpo_scene(row)
        if scene is None and "regular" in scenes:
            scene = val_slice(row)  # "regular" for ordinary non-empty positives
        if scene is None or scene not in scenes:
            continue
        meta = dict(_meta(row))
        meta["grpo_scene"] = scene
        meta["is_negative"] = bool(meta.get("is_negative")) or not assistant_text(row).strip()
        out = dict(row)
        out["metadata"] = meta
        by_scene.setdefault(scene, []).append(out)

    rng = random.Random(int(seed))
    for bucket in by_scene.values():
        rng.shuffle(bucket)
    if not max_samples or max_samples <= 0:
        picked: list[dict[str, Any]] = []
        for scene in sorted(by_scene):
            picked.extend(by_scene[scene])
        return picked
    if scene_mix:
        weights = {s: float(w) for s, w in scene_mix.items() if s in by_scene and float(w) > 0}
        quotas = quota_from_frac(weights, int(max_samples)) if weights else {}
        picked = []
        for scene in sorted(by_scene):
            take = int(quotas.get(scene, 0))
            if take:
                picked.extend(by_scene[scene][:take])
        rng.shuffle(picked)
        return picked[: int(max_samples)]
    counts = {scene: len(bucket) for scene, bucket in by_scene.items()}
    if not counts:
        return []
    quotas = quota_from_frac(counts, int(max_samples))
    picked = []
    for scene in sorted(by_scene):
        picked.extend(by_scene[scene][: int(quotas.get(scene, 0))])
    rng.shuffle(picked)
    return picked[: int(max_samples)]


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id")
            if sid:
                done.add(str(sid))
    return done


def generate_chunk(
    model,
    tokenizer,
    image,
    prompt: str,
    *,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    """Sample ``n`` completions from one image/prompt (HF num_return_sequences)."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    input_text = apply_point_chat_template(tokenizer, messages)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt")
    device = next(model.parameters()).device
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=max(1e-5, float(temperature)),
        top_p=min(1.0, max(0.01, float(top_p))),
        num_return_sequences=int(n),
        use_cache=True,
    )
    eos_ids = point_eos_token_ids(tokenizer)
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    pad_id = getattr(getattr(tokenizer, "tokenizer", tokenizer), "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else None
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = int(pad_id)
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    decoded: list[str] = []
    for i in range(out_ids.shape[0]):
        gen = out_ids[i][prompt_len:]
        raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
        decoded.append(cleaned_pred(raw))
    return decoded


def sample_group(
    model,
    tokenizer,
    image,
    prompt: str,
    *,
    g: int,
    gen_batch: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[list[str], int]:
    """Return G completions; shrink gen_batch on CUDA OOM. Returns (preds, used_batch)."""
    import torch

    preds: list[str] = []
    batch = max(1, int(gen_batch))
    while len(preds) < g:
        n = min(batch, g - len(preds))
        try:
            chunk = generate_chunk(
                model,
                tokenizer,
                image,
                prompt,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch <= 1:
                raise
            batch = max(1, batch // 2)
            print(f"OOM → gen_batch={batch}", flush=True)
            continue
        preds.extend(chunk)
    return preds[:g], batch


def write_summary(path: Path, rows: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_verdict = Counter()
    for rec in rows:
        by_scene[str(rec.get("scene") or "?")].append(rec)
        by_verdict[str(rec.get("verdict") or "?")] += 1

    def _scene_block(items: list[dict[str, Any]]) -> dict[str, Any]:
        stds = [float(r["reward_std"]) for r in items]
        means = [float(r["reward_mean"]) for r in items]
        n = len(items) or 1
        vc = Counter(str(r.get("verdict")) for r in items)
        return {
            "n": len(items),
            "frac_signal": vc.get("signal", 0) / n,
            "mean_of_means": sum(means) / n,
            "mean_of_stds": sum(stds) / n,
            "verdicts": dict(vc),
        }

    payload = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "n_scored": len(rows),
        "verdicts": dict(by_verdict),
        "frac_signal": by_verdict.get("signal", 0) / max(1, len(rows)),
        "overall": _scene_block(rows) if rows else {},
        "by_scene": {k: _scene_block(v) for k, v in sorted(by_scene.items())},
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=ROOT / "data/splits_stage_q1_withreal/train.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints/grpo_value_probe")
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--adapter", type=Path, default=ROOT / "checkpoints/q1_withreal/adapter_final")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    ap.add_argument(
        "--reward-set",
        choices=["q1", "q1v2", "q2"],
        default="q1",
        help="Reward used to score the G rollouts (match the training --reward-set).",
    )
    ap.add_argument("--include-empty", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument(
        "--gen-batch",
        type=int,
        default=8,
        help="Completions per generate() call. Same image repeated. OOM halves this.",
    )
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--max-pixels", type=int, default=2880 * 2880)
    ap.add_argument("--min-pixels", type=int, default=448 * 448)
    ap.add_argument(
        "--scene-mix",
        default="",
        help="Cap allocation across scenes, e.g. 'empty:0.4,multi_frag:0.25,semantic_group:0.25,regular:0.1'. "
        "Empty = proportional to availability.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-std", type=float, default=0.04)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    scenes = {s.strip() for s in args.scenes.split(",") if s.strip()}
    if args.include_empty:
        scenes.add("empty")
    reward_fn = _reward_fn(args.reward_set)
    scene_mix: dict[str, float] = {}
    for part in str(args.scene_mix or "").split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                scene_mix[k.strip()] = float(v)
            except ValueError:
                pass

    src_rows = load_jsonl(args.src)
    rows = select_probe_rows(
        src_rows, scenes=scenes, max_samples=args.max_samples, seed=args.seed, scene_mix=scene_mix or None
    )
    if not rows:
        raise SystemExit(f"no GRPO-eligible rows from {args.src} scenes={sorted(scenes)}")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"
    done_ids = _load_done_ids(rows_path)
    pending = []
    for row in rows:
        sid = str(_meta(row).get("sample_id") or "")
        if sid and sid in done_ids:
            continue
        pending.append(row)

    run_cfg = {
        "started_at": datetime.now().astimezone().isoformat(),
        "src": str(args.src),
        "n_eligible": len(rows),
        "n_already_done": len(done_ids),
        "n_pending": len(pending),
        "scenes": sorted(scenes),
        "g": args.g,
        "gen_batch": args.gen_batch,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "adapter": str(args.adapter),
        "merged_cache": str(MERGED_SFT_CACHE),
        "max_pixels": args.max_pixels,
        "min_std": args.min_std,
        "note": "signal = group std>=min_std and >=2 unique completions. "
        "saturated/collapsed/flat have no GRPO advantage.",
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"probe eligible={len(rows)} pending={len(pending)} done={len(done_ids)} "
        f"G={args.g} gen_batch={args.gen_batch} T={args.temperature} → {out_dir}",
        flush=True,
    )
    if not pending:
        scored = load_jsonl(rows_path) if rows_path.is_file() else []
        write_summary(summary_path, scored, {"status": "already_complete"})
        print(f"nothing to do; summary → {summary_path}", flush=True)
        return

    class _Args:
        adapter = args.adapter
        model = args.model
        no_4bit = args.no_4bit
        max_seq_length = args.max_seq_length

    import torch
    from unsloth import FastVisionModel

    _quiet_unsloth_noise()
    model, tokenizer, _peft = _load_policy(_Args())
    FastVisionModel.for_inference(model)
    model.eval()
    gen_batch = int(args.gen_batch)

    scored_live: list[dict[str, Any]] = load_jsonl(rows_path) if rows_path.is_file() else []
    t0 = time.time()
    peak_mem = 0.0

    with rows_path.open("a", encoding="utf-8") as fout:
        for i, row in enumerate(pending):
            meta = _meta(row)
            sid = str(meta.get("sample_id") or f"row{i}")
            images = row.get("images") or []
            if not images:
                continue
            from PIL import Image

            img = Image.open(str(images[0])).convert("RGB")
            img = resize_for_ovis(
                img, min_pixels=args.min_pixels, max_pixels=args.max_pixels
            )
            prompt = user_text(row).strip()
            target = assistant_text(row)
            vis = _vision_tokens(
                img.size[1], img.size[0], min_pixels=args.min_pixels, max_pixels=args.max_pixels
            )
            if args.seed is not None:
                torch.manual_seed(int(args.seed) + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(args.seed) + i)
            preds, gen_batch = sample_group(
                model,
                tokenizer,
                img,
                prompt,
                g=int(args.g),
                gen_batch=gen_batch,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            rewards = reward_fn(
                preds, target, is_negative=bool(meta.get("is_negative"))
            )
            stats = group_reward_stats(rewards)
            unique = list(dict.fromkeys(preds))
            verdict = group_learning_verdict(
                stats["mean"], stats["std"], len(unique), min_std=args.min_std
            )
            rec = {
                "sample_id": sid,
                "scene": meta.get("grpo_scene"),
                "pool_id": meta.get("pool_id"),
                "is_negative": bool(meta.get("is_negative")),
                "vis_tokens": vis,
                "target": target,
                "completions": preds,
                "rewards": rewards,
                "reward_mean": stats["mean"],
                "reward_std": stats["std"],
                "reward_min": stats["min"],
                "reward_max": stats["max"],
                "n_unique": len(unique),
                "verdict": verdict,
                "gen_batch": gen_batch,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            scored_live.append(rec)
            if torch.cuda.is_available():
                peak_mem = max(
                    peak_mem, torch.cuda.max_memory_allocated() / (1024**3)
                )
            done_n = len(scored_live)
            if done_n % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(pending) - i - 1) / max(rate, 1e-6)
                write_summary(
                    summary_path,
                    scored_live,
                    {
                        "status": "running",
                        "pending_left": len(pending) - i - 1,
                        "sec_per_prompt": elapsed / (i + 1),
                        "eta_sec": eta,
                        "peak_cuda_gb": peak_mem,
                        "gen_batch": gen_batch,
                    },
                )
                print(
                    f"[{done_n}/{len(rows)}] {sid} scene={rec['scene']} "
                    f"mean={stats['mean']:.3f} std={stats['std']:.3f} "
                    f"uniq={len(unique)} {verdict} vis={vis} "
                    f"batch={gen_batch} {rate:.2f} p/s eta={eta/60:.1f}m",
                    flush=True,
                )

    write_summary(
        summary_path,
        scored_live,
        {
            "status": "done",
            "duration_sec": time.time() - t0,
            "peak_cuda_gb": peak_mem,
            "gen_batch": gen_batch,
        },
    )
    print(f"done. summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()

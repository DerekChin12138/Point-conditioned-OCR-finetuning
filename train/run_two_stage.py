#!/usr/bin/env python3
"""Run Stage A1 then A2 in one process chain.

A2 --adapter is the absolute path of THIS session's A1 adapter_final.
It is never discovered by globbing checkpoints/. A2 will not start unless
that directory contains adapter_config.json + LoRA weights written after A1 began.

Hyperparams: train/curriculum.yaml (CLI flags override).

  bash train/run_two_stage.sh
  bash train/run_two_stage.sh --wait-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train" / "unsloth_stage_a.py"
CURRICULUM = ROOT / "train" / "curriculum.yaml"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt")


def _load_curriculum(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"curriculum {path} is not a mapping")
    return data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(adapter: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for p in sorted(adapter.iterdir()):
        if p.is_file():
            files[p.name] = {"size": p.stat().st_size, "sha256": _sha256(p), "mtime": p.stat().st_mtime}
    return files


def require_peft_adapter(path: Path, *, min_mtime: float | None = None) -> Path:
    """Fail unless ``path`` is a real PEFT adapter dir, optionally written after ``min_mtime``."""
    path = path.resolve()
    if not path.is_dir():
        raise SystemExit(f"adapter dir missing: {path}")
    cfg = path / "adapter_config.json"
    if not cfg.is_file():
        raise SystemExit(f"not a PEFT adapter (missing adapter_config.json): {path}")
    if not any((path / name).is_file() for name in ADAPTER_WEIGHTS):
        raise SystemExit(f"adapter has config but no LoRA weights in {ADAPTER_WEIGHTS}: {path}")
    if min_mtime is not None:
        newest = max(p.stat().st_mtime for p in path.iterdir() if p.is_file())
        if newest < min_mtime - 2.0:
            raise SystemExit(
                f"refusing stale adapter (files older than this A1 run): {path} "
                f"newest_mtime={newest} a1_start={min_mtime}"
            )
    peft_cfg = json.loads(cfg.read_text(encoding="utf-8"))
    print(
        f"[adapter-ok] {path}\n"
        f"  r={peft_cfg.get('r')} lora_alpha={peft_cfg.get('lora_alpha')} "
        f"peft_type={peft_cfg.get('peft_type')}",
        flush=True,
    )
    return path


def _wait_file(path: Path, *, timeout_sec: float, poll_sec: float) -> None:
    t0 = time.time()
    while not path.is_file():
        elapsed = time.time() - t0
        if elapsed > timeout_sec:
            raise SystemExit(f"timed out waiting for {path} ({timeout_sec:.0f}s)")
        print(f"[wait] {path} missing  t={elapsed:.0f}s", flush=True)
        time.sleep(poll_sec)
    print(f"[wait] ready {path}", flush=True)


def _run_stage(tag: str, argv: list[str]) -> None:
    print(f"\n========== {tag} ==========", flush=True)
    print(" ".join(argv), flush=True)
    proc = subprocess.run(argv, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"{tag} failed with exit {proc.returncode} — not starting later stages")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curriculum", type=Path, default=CURRICULUM)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--out-root", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--session", default=None)
    ap.add_argument("--wait-data", action="store_true")
    ap.add_argument("--wait-timeout-sec", type=float, default=8 * 3600)
    ap.add_argument("--wait-poll-sec", type=float, default=30)
    ap.add_argument("--skip-post-eval", action="store_true")
    args = ap.parse_args()

    cur = _load_curriculum(args.curriculum)
    a1c = dict(cur.get("a1") or {})
    a2c = dict(cur.get("a2") or {})
    a1_data = ROOT / a1c["data"]
    a1_val = ROOT / a1c["val"]
    a2_data = ROOT / a2c["data"]
    a2_val = ROOT / a2c["val"]

    if args.wait_data:
        for p in (a1_data, a1_val, a2_data, a2_val):
            _wait_file(p, timeout_sec=args.wait_timeout_sec, poll_sec=args.wait_poll_sec)
    for p in (a1_data, a1_val, a2_data, a2_val):
        if not p.is_file():
            raise SystemExit(f"missing split {p} (compose first, or pass --wait-data)")

    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S_curriculum")
    a1_dir = (args.out_root / f"{session}_a1").resolve()
    a2_dir = (args.out_root / f"{session}_a2").resolve()
    adapter_name = str(a2c.get("adapter_from") or "adapter_final")
    planned_adapter = a1_dir / adapter_name
    if a1_dir.exists() and any(a1_dir.iterdir()):
        raise SystemExit(f"A1 run dir already exists, refusing to reuse: {a1_dir}")
    if planned_adapter.exists():
        raise SystemExit(f"refusing pre-existing adapter (would be ambiguous): {planned_adapter}")

    py = str(args.python)
    skip_post = bool(args.skip_post_eval or cur.get("skip_post_eval"))

    def _pick(stage: dict[str, Any], key: str, default: Any = None) -> Any:
        if key in stage and stage[key] is not None:
            return stage[key]
        return cur[key] if key in cur else default

    def _stage_cmd(stage: dict[str, Any], *, for_continue: bool) -> list[str]:
        cmd = [
            py,
            "-u",
            str(TRAIN_PY),
            "--model",
            str(cur["model"]),
            "--lora-rank",
            str(cur["lora_rank"]),
            "--lora-alpha",
            str(cur["lora_alpha"]),
            "--epochs",
            str(_pick(stage, "epochs")),
            "--batch-size",
            str(_pick(stage, "batch_size")),
            "--grad-accum",
            str(_pick(stage, "grad_accum")),
            "--early-stopping-patience",
            str(cur["early_stopping_patience"]),
            "--save-steps",
            str(cur["save_steps"]),
            "--eval-steps",
            str(cur["eval_steps"]),
            "--logging-steps",
            str(cur["logging_steps"]),
            "--eval-batch-size",
            str(cur["eval_batch_size"]),
            "--seed",
            str(cur["seed"]),
            "--warmup-ratio",
            str(cur.get("warmup_ratio", 0.03)),
            "--weight-decay",
            str(cur.get("weight_decay", 0.01)),
            "--lr-scheduler",
            str(cur.get("lr_scheduler", "cosine")),
            "--optim",
            str(cur.get("optim", "adamw_8bit")),
            "--max-grad-norm",
            str(cur.get("max_grad_norm", 1.0)),
            "--max-seq-length",
            str(cur["max_seq_length"]),
            "--max-pixels",
            str(cur["max_pixels"]),
            "--min-pixels",
            str(cur["min_pixels"]),
        ]
        if cur.get("no_4bit", True):
            cmd.append("--no-4bit")
        if cur.get("finetune_vision", True):
            cmd.append("--finetune-vision")
        if skip_post:
            cmd.append("--skip-post-eval")
        post_n = int(cur.get("post_eval_max") or 0)
        if post_n > 0:
            cmd.extend(["--post-eval-max", str(post_n)])
        if for_continue:
            print(
                "[note] A2 LoRA r/α come from A1 adapter_final; "
                f"A1 created r={cur['lora_rank']} α={cur['lora_alpha']}. "
                "A requested A2 rank change cannot be applied on continue-train.",
                flush=True,
            )
        return cmd

    plan = {
        "session": session,
        "curriculum": str(args.curriculum.resolve()),
        "a1": {
            "out": str(a1_dir),
            "n": a1c.get("n"),
            "mix": a1c.get("mix"),
            "data": str(a1_data),
            "val": str(a1_val),
            "lr": a1c.get("lr"),
            "epochs": _pick(a1c, "epochs"),
            "batch_size": _pick(a1c, "batch_size"),
            "grad_accum": _pick(a1c, "grad_accum"),
            "effective_batch": int(_pick(a1c, "batch_size")) * int(_pick(a1c, "grad_accum")),
            "lora_rank": cur.get("lora_rank"),
            "lora_alpha": cur.get("lora_alpha"),
            "adapter": None,
        },
        "a2": {
            "out": str(a2_dir),
            "n": a2c.get("n"),
            "mix": a2c.get("mix"),
            "data": str(a2_data),
            "val": str(a2_val),
            "lr": a2c.get("lr"),
            "epochs": _pick(a2c, "epochs"),
            "batch_size": _pick(a2c, "batch_size"),
            "grad_accum": _pick(a2c, "grad_accum"),
            "effective_batch": int(_pick(a2c, "batch_size")) * int(_pick(a2c, "grad_accum")),
            "lora_rank": "inherit_from_a1_adapter",
            "lora_alpha": "inherit_from_a1_adapter",
            "adapter": str(planned_adapter),
        },
        "shared": {k: cur[k] for k in (
            "model", "no_4bit", "finetune_vision", "lora_rank", "lora_alpha",
            "epochs", "batch_size", "grad_accum", "early_stopping_patience",
            "save_steps", "eval_steps", "seed", "max_seq_length", "max_pixels",
            "min_pixels", "warmup_ratio", "weight_decay", "lr_scheduler", "optim",
            "max_grad_norm",
        ) if k in cur},
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

    a1_started = time.time()
    _run_stage(
        "A1",
        [
            *_stage_cmd(a1c, for_continue=False),
            "--data",
            str(a1_data),
            "--val",
            str(a1_val),
            "--lr",
            str(a1c["lr"]),
            "--out",
            str(a1_dir),
        ],
    )

    adapter = require_peft_adapter(planned_adapter, min_mtime=a1_started)
    adapter_abs = str(adapter)
    handoff = {
        "session": session,
        "a1_run_dir": str(a1_dir),
        "adapter_final": adapter_abs,
        "peft": json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8")),
        "files": _fingerprint(adapter),
        "a1_started_unix": a1_started,
        "handoff_at": datetime.now().astimezone().isoformat(),
    }
    handoff_path = a1_dir / "handoff_to_a2.json"
    handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    print(f"[handoff] wrote {handoff_path}", flush=True)
    print(f"[handoff] A2 --adapter {adapter_abs}", flush=True)

    a2_argv = [
        *_stage_cmd(a2c, for_continue=True),
        "--data",
        str(a2_data),
        "--val",
        str(a2_val),
        "--lr",
        str(a2c["lr"]),
        "--adapter",
        adapter_abs,
        "--out",
        str(a2_dir),
    ]
    if "--adapter" not in a2_argv or adapter_abs not in a2_argv:
        raise SystemExit("internal error: A2 argv missing exact adapter path")
    idx = a2_argv.index("--adapter")
    if Path(a2_argv[idx + 1]).resolve() != adapter:
        raise SystemExit("internal error: A2 --adapter does not match A1 adapter_final")

    _run_stage("A2", a2_argv)
    print(f"[done] A1={a1_dir}\n       A2={a2_dir}\n       used_adapter={adapter_abs}", flush=True)


if __name__ == "__main__":
    main()

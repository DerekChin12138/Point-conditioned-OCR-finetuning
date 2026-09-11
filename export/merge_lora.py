#!/usr/bin/env python3
"""Merge LoRA adapter into base OvisOCR2 weights before GGUF conversion."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ATH-MaaS/OvisOCR2")
    ap.add_argument("--adapter", required=True, help="LoRA checkpoint dir")
    ap.add_argument("--out", required=True, help="Merged HF model dir")
    args = ap.parse_args()

    try:
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except Exception as e:
        raise SystemExit(
            f"Install transformers/peft on the CUDA export host. Import error: {e}"
        ) from e

    # Class name may differ for Qwen3.5-VL / OvisOCR2 — adjust after inspecting config.json
    print(f"Loading base {args.base} ...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(args.base, trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True)

    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    try:
        proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
        proc.save_pretrained(args.out)
    except Exception as e:
        print(f"[warn] processor save failed: {e}")
    print(f"Merged model → {args.out}")


if __name__ == "__main__":
    main()

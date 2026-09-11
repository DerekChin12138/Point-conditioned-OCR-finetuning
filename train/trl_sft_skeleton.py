#!/usr/bin/env python3
"""Optional TRL + PEFT QLoRA path when LLaMA-Factory template for OvisOCR2 is awkward.

This is a skeleton: wire processor/model classes after inspecting ATH-MaaS/OvisOCR2
config on a CUDA box. Prefer LLaMA-Factory if `qwen3_vl` (or the model's native
template) already works for this checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ATH-MaaS/OvisOCR2")
    ap.add_argument("--data", type=Path, required=True, help="ShareGPT JSON")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/trl_point"))
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    raise SystemExit(
        "TRL trainer skeleton: install train extras on CUDA host, then implement "
        "dataset mapping with the model's AutoProcessor / chat template. "
        f"Requested model={args.model} data={args.data} out={args.out} steps={args.max_steps}"
    )


if __name__ == "__main__":
    main()

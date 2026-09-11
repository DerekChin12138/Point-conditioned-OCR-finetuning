#!/usr/bin/env bash
# Stage A via Unsloth (recommended). Linux + NVIDIA only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-data/splits/train.jsonl}"
MODEL="${MODEL:-ATH-MaaS/OvisOCR2}"
OUT="${OUT:-checkpoints/stage_a_point_unsloth}"

exec uv run python train/unsloth_stage_a.py \
  --data "$DATA" \
  --model "$MODEL" \
  --out "$OUT" \
  "$@"

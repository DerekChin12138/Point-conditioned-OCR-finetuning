#!/usr/bin/env bash
# Stage A via Unsloth (recommended). Linux + NVIDIA only.
# Writes checkpoints/<YYYYMMDD_HHMMSS>_stage_a/ unless OUT is set.
#
# Defaults in train/unsloth_stage_a.py:
#   --batch-size 2 --grad-accum 4 --lora-rank 16 --lora-alpha 32
#   --max-pixels $((2880*2880)) --min-pixels $((448*448))
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-data/splits/train.jsonl}"
VAL="${VAL:-data/splits/val.jsonl}"
MODEL="${MODEL:-ATH-MaaS/OvisOCR2}"
OUT_ROOT="${OUT_ROOT:-checkpoints}"

# Optional: OUT=/path/to/exact_run_dir to skip auto timestamp
EXTRA=()
if [[ -n "${OUT:-}" ]]; then
  EXTRA+=(--out "$OUT")
fi

exec uv run python train/unsloth_stage_a.py \
  --data "$DATA" \
  --val "$VAL" \
  --model "$MODEL" \
  --out-root "$OUT_ROOT" \
  "${EXTRA[@]}" \
  "$@"

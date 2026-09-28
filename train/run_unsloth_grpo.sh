#!/usr/bin/env bash
# ARCHIVED (2026-09-26): the 0.8B GRPO artifacts this referenced (grpo_q1_signal,
# q1_withreal_merged, grpo_q2_marker) were deleted. Kept only as a historical
# recipe reference. Do not run without pointing MODEL/ADAPTER/DATA at live paths.
#
# GRPO_2: continue training grpo_q1_signal LoRA on dual-size marker split.
# π_θ = existing adapter (trainable); KL disable_adapter() → q1_withreal_merged.
# Writes checkpoints/grpo_q2_marker/ unless RUN_ID / OUT is set.
# Bake+new-LoRA mode: CONTINUE_ADAPTER=0 bash train/run_unsloth_grpo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-data/splits_grpo_q2_marker/train.jsonl}"
VAL="${VAL:-data/splits_grpo_q2_marker/val.jsonl}"
MODEL="${MODEL:-models/Qwen3.5-0.8B}"
ADAPTER="${ADAPTER:-checkpoints/grpo_q1_signal/adapter_final}"
OUT_ROOT="${OUT_ROOT:-checkpoints}"
RUN_ID="${RUN_ID:-grpo_q2_marker}"
CONTINUE_ADAPTER="${CONTINUE_ADAPTER:-1}"

EXTRA=()
if [[ -n "${OUT:-}" ]]; then
  EXTRA+=(--out "$OUT")
fi
if [[ "$CONTINUE_ADAPTER" == "1" || "$CONTINUE_ADAPTER" == "true" ]]; then
  EXTRA+=(--continue-adapter)
fi

exec uv run python train/unsloth_grpo.py \
  --data "$DATA" \
  --val "$VAL" \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --out-root "$OUT_ROOT" \
  --run-id "$RUN_ID" \
  "${EXTRA[@]}" \
  "$@"

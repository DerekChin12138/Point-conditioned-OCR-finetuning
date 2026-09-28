#!/usr/bin/env bash
# ARCHIVED (2026-09-26): the 0.8B artifacts this referenced (checkpoints/q1_grpo_merged)
# were deleted. Kept only as a historical recipe reference.
# Use train/run_08b_pipeline.sh for the current 0.8B run, or the command in
# docs/PIPELINE_QWEN35_2B.md §7 for 2B.
#
# SFT_Q2 OCR+MT — do not start this unless you mean to train.
# Recipe: more conservative than Q1 SFT, more aggressive than GRPO.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# WSL: fused CE must not probe leftover VRAM (shared GPU RAM is not CUDA free).
export UNSLOTH_CE_LOSS_N_CHUNKS="${UNSLOTH_CE_LOSS_N_CHUNKS:-32}"

uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q2_ocr_mt/train.jsonl \
  --val data/splits_stage_q2_ocr_mt/val.jsonl \
  --model checkpoints/q1_grpo_merged \
  --out-root checkpoints \
  --run-id q2_ocr_mt \
  --max-seq-length 4096 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 2e-5 \
  --epochs 1.0 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --save-steps 500 \
  --eval-steps 250 \
  --logging-steps 10 \
  --eval-batch-size 2 \
  --num-workers 4 \
  --seed 42 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --lr-scheduler cosine \
  --optim adamw_8bit \
  --max-grad-norm 1.0

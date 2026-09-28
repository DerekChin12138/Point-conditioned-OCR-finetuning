#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-0.8B 两轮 SFT（对齐 2B 的当前数据配方 / 准星协议 / 序列长度）
#
#   bash train/run_08b_pipeline.sh smoke       # 只加载 + 2 步，验证脚本（不写正式 run）
#   CONFIRM=1 bash train/run_08b_pipeline.sh q1        # SFT_Q1
#   bash train/run_08b_pipeline.sh merge-q1            # 烘焙 Q1 整模
#   CONFIRM=1 bash train/run_08b_pipeline.sh q2        # SFT_Q2（需先 merge-q1）
#   bash train/run_08b_pipeline.sh merge-q2            # 烘焙 Q2 整模（可部署）
#   bash train/run_08b_pipeline.sh eval-q1             # Q1 test 评测
#   bash train/run_08b_pipeline.sh eval-q2             # Q2 val 拆分指标
#   CONFIRM=1 bash train/run_08b_pipeline.sh all       # q1 → merge-q1 → q2 → merge-q2 → evals
#
# 对齐口径（与 checkpoints/q1_2b / q2_2b 完全一致，除开基座本身）：
#   - 数据：data/splits_stage_q1_withreal/ + data/splits_stage_q2_ocr_mt_v2/
#   - 准星：x45r（0.08%–0.2% 面积，已在数据里）
#   - seq：8192（不要用 4096：最密页 ~8000 token）
#   - 4bit QLoRA r32/α64；Q1 训 vision，Q2 冻 vision
#   - Q1: lr 5e-5, 1.5 epoch | Q2: lr 1.5e-5, 1.0 epoch
#
# 与旧 0.8B run（checkpoints/q1_withreal / q2_ocr_mt）的差异（这就是为什么要重跑）：
#   旧 Q1: seq 4096, bf16, 旧大准星(x45c/x45d) | 旧 Q2: seq 4096, 未清洗负例,
#   底模 = 已证伪的 q1_grpo_merged, --finetune-vision 还是 no-op bug
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- 可覆盖的环境变量 --------------------------------------------------------
MODEL="${MODEL:-models/Qwen3.5-0.8B}"
OUT_ROOT="${OUT_ROOT:-checkpoints}"
RUN_Q1="${RUN_Q1:-q1_08b}"
RUN_Q2="${RUN_Q2:-q2_08b}"
MERGED_Q1="${MERGED_Q1:-${OUT_ROOT}/${RUN_Q1}_merged}"
MERGED_Q2="${MERGED_Q2:-${OUT_ROOT}/${RUN_Q2}_merged}"
Q1_TRAIN="${Q1_TRAIN:-data/splits_stage_q1_withreal/train.jsonl}"
Q1_VAL="${Q1_VAL:-data/splits_stage_q1_withreal/val.jsonl}"
Q1_TEST="${Q1_TEST:-data/splits_stage_q1_withreal/test.jsonl}"
Q2_TRAIN="${Q2_TRAIN:-data/splits_stage_q2_ocr_mt_v2/train.jsonl}"
Q2_VAL="${Q2_VAL:-data/splits_stage_q2_ocr_mt_v2/val.jsonl}"
# 精度：4bit（与 2B 对齐）| bf16（0.8B 显存足够，质量略好，但不是与 2B 的等效比较）
PRECISION="${PRECISION:-4bit}"
# 宿主 15GB：实测 `--pin-memory` + `--persistent-workers` 在长跑里会让宿主内存缓慢上涨
# （2026-09-26：MemAvailable 2.5GB → 0.7GB，被看门狗击杀）。默认改为 workers=4、不用 pin、
# 不用 persistent、prefetch=2；吞吐/内存的平衡点。confirm 过的更激进配置可用 PIN_MEMORY=1 打开。
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH="${PREFETCH:-2}"
PIN_MEMORY="${PIN_MEMORY:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
# 断点续训：RESUME_Q1 / RESUME_Q2 指向 checkpoint-N → 传 --resume-from-checkpoint
RESUME_Q1="${RESUME_Q1:-}"
RESUME_Q2="${RESUME_Q2:-}"
# 分辨率/序列：Qwen3.5 视觉 token ≈ max_pixels/1024（patch16×merge2）。
# 2026-09-28 之前 collator 默认 resize="min" 会把图像宽度静默压到 768（≈336 token），
# `--max-pixels` 形同虚设；现在默认 resize="max"，由下面变量控制。
#   1280²=1.64MP → ~1600 vis-tok（省钱）
#   2048²=4.19MP → ~4200 vis-tok（默认；最密页 peak ~3.4GB，seq 8192 够）
#   2880²=8.29MP → ~8100 vis-tok（需 MAX_SEQ_LENGTH ≥ 12288，peak ~5.8GB）
MAX_PIXELS="${MAX_PIXELS:-4194304}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
COLLATOR_RESIZE="${COLLATOR_RESIZE:-max}"
# 超参（默认 = 0.8B 本机配方；32GB 卡/4B 见 docs/CLOUD_RUN_32GB.md）
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR_Q1="${LR_Q1:-5e-5}"
LR_Q2="${LR_Q2:-1.5e-5}"
EPOCHS_Q1="${EPOCHS_Q1:-1.5}"
EPOCHS_Q2="${EPOCHS_Q2:-1.0}"
EVAL_BATCH_Q1="${EVAL_BATCH_Q1:-2}"
EVAL_BATCH_Q2="${EVAL_BATCH_Q2:-1}"
POST_EVAL_BATCH_Q1="${POST_EVAL_BATCH_Q1:-8}"
POST_EVAL_BATCH_Q2="${POST_EVAL_BATCH_Q2:-2}"
SMOKE_OUT="${SMOKE_OUT:-/tmp/point_ocr_smoke_08b}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# WSL：fused CE 不要探测残余显存（共享显存不是 CUDA free）
export UNSLOTH_CE_LOSS_N_CHUNKS="${UNSLOTH_CE_LOSS_N_CHUNKS:-32}"

PREC_FLAGS=()
if [[ "$PRECISION" == "bf16" ]]; then
  PREC_FLAGS+=(--no-4bit)
fi
LOADER_FLAGS=()
if [[ "$NUM_WORKERS" -gt 0 ]]; then
  LOADER_FLAGS+=(--num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH")
  if [[ "$PERSISTENT_WORKERS" == "1" ]]; then LOADER_FLAGS+=(--persistent-workers); else LOADER_FLAGS+=(--no-persistent-workers); fi
  if [[ "$PIN_MEMORY" == "1" ]]; then LOADER_FLAGS+=(--pin-memory); else LOADER_FLAGS+=(--no-pin-memory); fi
else
  LOADER_FLAGS+=(--num-workers 0 --no-pin-memory --no-persistent-workers)
fi

RESUME_FLAGS_Q1=()
[[ -n "$RESUME_Q1" ]] && RESUME_FLAGS_Q1+=(--resume-from-checkpoint "$RESUME_Q1")
RESUME_FLAGS_Q2=()
[[ -n "$RESUME_Q2" ]] && RESUME_FLAGS_Q2+=(--resume-from-checkpoint "$RESUME_Q2")

require_confirm() {
  if [[ "${CONFIRM:-0}" != "1" ]]; then
    echo "refusing to start a long training run without CONFIRM=1" >&2
    echo "  re-run: CONFIRM=1 bash train/run_08b_pipeline.sh $1" >&2
    exit 1
  fi
}

check_base() {
  local d="$1"
  if [[ ! -f "$d/config.json" ]]; then
    echo "base not found (need config.json): $d" >&2
    exit 1
  fi
}

stage_q1() {
  check_base "$MODEL"
  echo "== [q1] base=$MODEL out=$OUT_ROOT/$RUN_Q1 precision=$PRECISION workers=$NUM_WORKERS"
  uv run python train/unsloth_stage_a.py \
    --data "$Q1_TRAIN" \
    --val  "$Q1_VAL" \
    --model "$MODEL" \
    --out-root "$OUT_ROOT" --run-id "$RUN_Q1" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
    --lr "$LR_Q1" --epochs "$EPOCHS_Q1" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --finetune-vision \
    --save-steps 250 --eval-steps 250 --logging-steps 10 --eval-batch-size "$EVAL_BATCH_Q1" \
    "${LOADER_FLAGS[@]}" --tf32 \
    --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
    --optim adamw_8bit --max-grad-norm 1.0 \
    --max-pixels "$MAX_PIXELS" --min-pixels "$MIN_PIXELS" \
    --collator-resize "$COLLATOR_RESIZE" \
    --post-eval-batch-size "$POST_EVAL_BATCH_Q1" --seed 42 \
    "${PREC_FLAGS[@]}" "${RESUME_FLAGS_Q1[@]}"
}

stage_merge_q1() {
  check_base "$MODEL"
  echo "== [merge-q1] $MODEL + $OUT_ROOT/$RUN_Q1/adapter_final -> $MERGED_Q1"
  uv run python export/merge_unsloth_lora.py \
    --base "$MODEL" \
    --adapter "$OUT_ROOT/$RUN_Q1/adapter_final" \
    --out "$MERGED_Q1" \
    --max-seq-length "$MAX_SEQ_LENGTH"
}

stage_q2() {
  check_base "$MERGED_Q1"
  echo "== [q2] base=$MERGED_Q1 out=$OUT_ROOT/$RUN_Q2 precision=$PRECISION workers=$NUM_WORKERS"
  uv run python train/unsloth_stage_a.py \
    --data "$Q2_TRAIN" \
    --val  "$Q2_VAL" \
    --model "$MERGED_Q1" \
    --out-root "$OUT_ROOT" --run-id "$RUN_Q2" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
    --lr "$LR_Q2" --epochs "$EPOCHS_Q2" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --save-steps 250 --eval-steps 250 --logging-steps 10 --eval-batch-size "$EVAL_BATCH_Q2" \
    "${LOADER_FLAGS[@]}" --tf32 \
    --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
    --optim adamw_8bit --max-grad-norm 1.0 \
    --max-pixels "$MAX_PIXELS" --min-pixels "$MIN_PIXELS" \
    --collator-resize "$COLLATOR_RESIZE" \
    --post-eval-batch-size "$POST_EVAL_BATCH_Q2" --seed 42 \
    "${PREC_FLAGS[@]}" "${RESUME_FLAGS_Q2[@]}"
}

stage_merge_q2() {
  check_base "$MERGED_Q1"
  echo "== [merge-q2] $MERGED_Q1 + $OUT_ROOT/$RUN_Q2/adapter_final -> $MERGED_Q2"
  uv run python export/merge_unsloth_lora.py \
    --base "$MERGED_Q1" \
    --adapter "$OUT_ROOT/$RUN_Q2/adapter_final" \
    --out "$MERGED_Q2" \
    --max-seq-length "$MAX_SEQ_LENGTH"
}

stage_eval_q1() {
  echo "== [eval-q1] adapter on test split"
  uv run python eval/run_unsloth_adapter_eval.py \
    --model "$MODEL" \
    --adapter "$OUT_ROOT/$RUN_Q1/adapter_final" \
    --data "$Q1_TEST" \
    --gen-batch-size 8 \
    --max-pixels "$MAX_PIXELS" --min-pixels "$MIN_PIXELS" \
    --out-dir "$OUT_ROOT/$RUN_Q1/eval_test" --prefix test
}

stage_eval_q2() {
  echo "== [eval-q2] split Q2 metrics on the post-eval predictions"
  uv run python eval/run_q2_eval.py \
    --predictions "$OUT_ROOT/$RUN_Q2/metrics/final_predictions.jsonl" \
    --split "$Q2_VAL" \
    --out-dir "$OUT_ROOT/$RUN_Q2/metrics" --prefix q2
}

stage_smoke() {
  check_base "$MODEL"
  echo "== [smoke] 8 samples / 2 steps, base=$MODEL, out=$SMOKE_OUT"
  rm -rf "$SMOKE_OUT"
  uv run python train/unsloth_stage_a.py \
    --data "$Q1_TRAIN" --val "$Q1_VAL" \
    --model "$MODEL" \
    --out "$SMOKE_OUT" \
    --max-seq-length "$MAX_SEQ_LENGTH" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --finetune-vision \
    --max-samples 8 --max-steps 2 --skip-post-eval \
    --num-workers 0 --no-pin-memory \
    --max-pixels "$MAX_PIXELS" --min-pixels "$MIN_PIXELS" \
    --collator-resize "$COLLATOR_RESIZE" \
    "${PREC_FLAGS[@]}"
  echo "== [smoke] ok — peak GPU 见 run_config.json"
}

case "${1:-}" in
  smoke)    stage_smoke ;;
  q1)       require_confirm q1;       stage_q1 ;;
  merge-q1) stage_merge_q1 ;;
  q2)       require_confirm q2;       stage_q2 ;;
  merge-q2) stage_merge_q2 ;;
  eval-q1)  stage_eval_q1 ;;
  eval-q2)  stage_eval_q2 ;;
  all)
    require_confirm all
    stage_q1; stage_merge_q1; stage_q2; stage_merge_q2
    stage_eval_q1 || true
    stage_eval_q2 || true
    ;;
  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac

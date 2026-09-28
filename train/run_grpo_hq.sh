#!/usr/bin/env bash
# =============================================================================
# GRPO HQ-200 一条龙（Q1 或 Q2 × 0.8B / 2B / 4B）
#
#   KIND=q1 SIZE=08b bash train/run_grpo_hq.sh status   # 前置检查（先跑这个）
#   KIND=q1 SIZE=08b bash train/run_grpo_hq.sh cand     # 候选池（弱点桶分层，~5s，确定性）
#   KIND=q1 SIZE=08b bash train/run_grpo_hq.sh seed     # 无 GPU：启发式 200/60（冒烟链路用）
#   KIND=q1 SIZE=08b GEN_BATCH=4 bash train/run_grpo_hq.sh probe    # G=8 探针（需要 GPU）
#   KIND=q1 SIZE=08b bash train/run_grpo_hq.sh compose  # 信号门禁 → 200/60
#   KIND=q1 SIZE=08b bash train/run_grpo_hq.sh dryrun   # 16 样本 / 2 步：验显存与参数
#   KIND=q1 SIZE=08b bash train/run_grpo_hq.sh train    # 正式 GRPO
#   KIND=q1 SIZE=08b bash train/run_grpo_hq.sh merge    # GRPO adapter → 整模
#   KIND=q1 SIZE=08b bash train/run_grpo_hq.sh eval     # SFT 基线 vs GRPO（同 val/batch/分辨率）
#   CONFIRM=1 KIND=q2 SIZE=4b GEN_BATCH=4 bash train/run_grpo_hq.sh all
#
# KIND=q1：点 OCR（a2_v3，只出块）  → reward q1v2
# KIND=q2：点 OCR + 翻译（ocr_mt_v1，XML）→ reward q2，默认 **定位优先权重**
#          GRPO_REWARD_WEIGHTS="1.5,0.05,1.0,1.5,1.5,0.75,0.5"
#          （source,translation,xml,empty,hallucination,over,leak；翻译权重 0.05≈不管质量）
#
# 硬约束：GRPO 必须与 SFT 同分辨率（默认 2880² / seq 12288）与同 prompt。
# =============================================================================
set -Eeuo pipefail
trap 'echo "[$(date +%T)] !!! ERROR at line $LINENO: $BASH_COMMAND"' ERR

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export UNSLOTH_CE_LOSS_N_CHUNKS=32
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1

log(){ echo; echo "==================== [$(date '+%F %T')] $* ===================="; }
gpu(){ nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"; }

# ---------------------------- 配置 -------------------------------------------
SIZE="${SIZE:-08b}"     # 08b | 2b | 4b
KIND="${KIND:-q1}"      # q1 | q2
case "$SIZE" in
  08b) RAW="models/Qwen3.5-0.8B"; Q1R="q1_08b"; Q2R="q2_08b" ;;
  2b)  RAW="models/Qwen3.5-2B";   Q1R="q1_2b";  Q2R="q2_2b" ;;
  4b)  RAW="models/Qwen3.5-4B";   Q1R="q1_4b";  Q2R="q2_4b" ;;
  *) echo "SIZE must be 08b|2b|4b"; exit 2 ;;
esac
case "$KIND" in
  q1)
    GRPO_MODEL="${GRPO_MODEL:-$RAW}"                                   # π 的底模
    GRPO_ADAPTER="${GRPO_ADAPTER:-checkpoints/$Q1R/adapter_final}"     # 烘成冻结 π_ref
    REF_MERGED="${REF_MERGED:-checkpoints/${Q1R}_merged}"              # 评测基线 / merge 底模
    CAND="${CAND:-data/splits_grpo_cand}"
    HQ="${HQ:-data/splits_grpo_q1_hq200}"
    REWARD_SET="q1v2"
    SCENES="regular,multi_frag,semantic_group"
    MAX_NEW="${MAX_NEW:-256}"
    ;;
  q2)
    GRPO_MODEL="${GRPO_MODEL:-checkpoints/${Q1R}_merged}"              # 先底模
    GRPO_ADAPTER="${GRPO_ADAPTER:-checkpoints/$Q2R/adapter_final}"     # 再 Q2 SFT LoRA
    REF_MERGED="${REF_MERGED:-checkpoints/${Q2R}_merged}"
    CAND="${CAND:-data/splits_grpo_cand_q2}"
    HQ="${HQ:-data/splits_grpo_q2_hq200}"
    REWARD_SET="q2"
    SCENES="regular,multi_frag,semantic_group"
    MAX_NEW="${MAX_NEW:-512}"
    # 定位优先：翻译几乎不计分（翻译质量交给数据/两阶段，不交给 RL）
    export GRPO_REWARD_WEIGHTS="${GRPO_REWARD_WEIGHTS:-1.5,0.05,1.0,1.5,1.5,0.75,0.5}"
    ;;
  *) echo "KIND must be q1|q2"; exit 2 ;;
esac
RUN_ID="${RUN_ID:-grpo_${KIND}_${SIZE}}"
PROBE_OUT="${PROBE_OUT:-checkpoints/grpo_value_probe_${KIND}_${SIZE}}"
SEQ="${SEQ:-12288}"; PX="${PX:-8294400}"; MINPX="${MINPX:-200704}"
G="${G:-8}"; GEN_BATCH="${GEN_BATCH:-1}"; BATCH="${BATCH:-2}"; ACCUM="${ACCUM:-8}"
LR="${LR:-5e-6}"; EPOCHS="${EPOCHS:-1.0}"; LORA_R="${LORA_R:-16}"; LORA_A="${LORA_A:-32}"; BETA="${BETA:-0.04}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"   # 探针抽样上限（0=全池 1500）；<池子时最终条数会变少
VAL_Q1="data/splits_stage_q1_withreal/val.jsonl"
VAL_Q2="data/splits_stage_q2_ocr_mt_v2/val.jsonl"

# ---------------------------- stages -----------------------------------------
stage_status(){
  log "[status] KIND=$KIND SIZE=$SIZE RUN_ID=$RUN_ID"
  echo "GRPO_MODEL   = $GRPO_MODEL   $([[ -f "$GRPO_MODEL/config.json" ]] && echo OK || echo MISSING)"
  echo "GRPO_ADAPTER = $GRPO_ADAPTER   $([[ -f "$GRPO_ADAPTER/adapter_config.json" ]] && echo OK || echo 'MISSING → 等 SFT 训完')"
  echo "REF_MERGED   = $REF_MERGED   $([[ -f "$REF_MERGED/config.json" ]] && echo OK || echo 'MISSING → 先 merge SFT')"
  echo "reward_set   = $REWARD_SET   MAX_NEW=$MAX_NEW   SEQ/PX=$SEQ/$PX    (必须与 SFT 一致)"
  [[ -n "${GRPO_REWARD_WEIGHTS:-}" ]] && echo "weights      = $GRPO_REWARD_WEIGHTS  (source,translation,xml,empty,halluc,over,leak)"
  echo "GPU          = $(gpu)"
  uv run python -u - "$CAND" "$HQ" "$PROBE_OUT" <<'PY'
import json, os, sys
cand, hq, probe = sys.argv[1:4]
p = os.path.join(hq, "split_meta.json")
if os.path.isfile(p):
    d = json.load(open(p, encoding="utf-8"))
    print(f"HQ 数据      = mode={d.get('mode')} kind={d.get('kind')} n_train={d.get('n_train')} "
          f"n_val={d.get('n_val')} reason={d.get('picked_by_reason')}")
    print("               mode=seed → 未做信号门禁（启发式）；verified 需先 probe 再 compose")
else:
    print(f"HQ 数据      = (无) 还没生成")
print(f"候选池       = {cand}/train.jsonl {'OK' if os.path.isfile(cand + '/train.jsonl') else '(无，先 cand)'}")
print(f"探针产物     = {probe}/rows.jsonl {'OK' if os.path.isfile(probe + '/rows.jsonl') else '(无)'}")
PY
}

stage_cand(){
  log "[cand] 建候选池（$KIND，1500/120，seed 42 确定性）"
  uv run python -u data/scripts/build_grpo_candidates.py --kind "$KIND"
}

stage_seed(){
  log "[seed] 无 GPU：启发式 200/60（signal 未验证，冒烟用）"
  uv run python -u data/scripts/compose_grpo_q1_hq200.py --kind "$KIND" --probe none
}

stage_probe(){
  log "[probe] $KIND / G=$G / GEN_BATCH=$GEN_BATCH / MAX_SAMPLES=$MAX_SAMPLES"
  [[ -f "$CAND/train.jsonl" ]] || { echo "缺 $CAND/train.jsonl —— 先 cand"; exit 1; }
  [[ -f "$GRPO_ADAPTER/adapter_config.json" ]] || { echo "缺 $GRPO_ADAPTER —— SFT 还没训完？"; exit 1; }
  uv run python -u eval/run_grpo_value_probe.py \
    --src "$CAND/train.jsonl" \
    --model "$GRPO_MODEL" --adapter "$GRPO_ADAPTER" \
    --out "$PROBE_OUT" --reward-set "$REWARD_SET" --scenes "$SCENES" \
    --g "$G" --gen-batch "$GEN_BATCH" --max-samples "$MAX_SAMPLES" \
    --temperature 1.2 --top-p 1.0 --max-new-tokens "$MAX_NEW" \
    --max-seq-length "$SEQ" --max-pixels "$PX" --min-pixels "$MINPX"
  cp -f "$PROBE_OUT/summary.json" "$PROBE_OUT/summary_$(date +%Y%m%d_%H%M%S).json" 2>/dev/null || true
}

stage_compose(){
  log "[compose] 信号门禁 → $HQ"
  [[ -f "$PROBE_OUT/rows.jsonl" ]] || { echo "缺探针产物，先 probe（或改用 seed）"; exit 1; }
  uv run python -u data/scripts/compose_grpo_q1_hq200.py --kind "$KIND" --probe "$PROBE_OUT/rows.jsonl"
  uv run python -u - "$HQ" <<'PY'
import collections, json, os, sys
hq = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(hq, "train.jsonl"), encoding="utf-8") if l.strip()]
meta = [r["metadata"] for r in rows]
d = json.load(open(os.path.join(hq, "split_meta.json"), encoding="utf-8"))
print("mode         :", d.get("mode"), "| probe_rows:", (d.get("probe_counts") or {}).get("probe_rows"),
      "| weights:", d.get("reward_weights_env"))
print("n / pages    :", len(rows), "/", len({m.get("page_id") for m in meta}))
print("reason       :", dict(collections.Counter(m.get("grp_reason") for m in meta)))
print("tertile      :", dict(sorted(collections.Counter(m.get("grp_tertile") for m in meta).items(), key=lambda kv: str(kv[0]))))
print("has signal   :", all("grpo_signal" in m for m in meta) if d.get("mode") == "probe" else "(seed: 未验证)")
print("images ok    :", all(os.path.exists(r["images"][0]) for r in rows))
PY
}

stage_dryrun(){
  log "[dryrun] 16 样本 / 2 步（看显存；不满意就 Ctrl-C 调 BATCH/GEN_BATCH）"
  uv run python -u train/unsloth_grpo.py \
    --data "$HQ/train.jsonl" --val "$HQ/val.jsonl" \
    --model "$GRPO_MODEL" --adapter "$GRPO_ADAPTER" \
    --out "$(mktemp -d /tmp/grpo_dry_XXXX)" --reward-set "$REWARD_SET" \
    --num-generations "$G" --gen-chunk-size 1 --max-completion-length "$MAX_NEW" \
    --batch-size "$BATCH" --grad-accum "$ACCUM" \
    --lora-rank "$LORA_R" --lora-alpha "$LORA_A" --lr "$LR" --epochs 1 \
    --max-samples 16 --max-steps 2 \
    --max-seq-length "$SEQ" --max-pixels "$PX" --min-pixels "$MINPX" \
    --skip-post-eval --seed 42
  echo "dryrun 通过。GPU: $(gpu)"
}

stage_train(){
  log "[train] → checkpoints/$RUN_ID"
  uv run python -u train/unsloth_grpo.py \
    --data "$HQ/train.jsonl" --val "$HQ/val.jsonl" \
    --model "$GRPO_MODEL" --adapter "$GRPO_ADAPTER" \
    --out-root checkpoints --run-id "$RUN_ID" --reward-set "$REWARD_SET" \
    --num-generations "$G" --gen-chunk-size 1 --max-completion-length "$MAX_NEW" \
    --batch-size "$BATCH" --grad-accum "$ACCUM" \
    --lora-rank "$LORA_R" --lora-alpha "$LORA_A" --lr "$LR" --epochs "$EPOCHS" --beta "$BETA" \
    --save-steps 25 --logging-steps 5 \
    --max-seq-length "$SEQ" --max-pixels "$PX" --min-pixels "$MINPX" \
    --num-workers 4 --max-grad-norm 1.0 --optim adamw_8bit --seed 42
}

stage_merge(){
  log "[merge] $REF_MERGED + $RUN_ID/adapter_final → checkpoints/${RUN_ID}_merged"
  uv run python -u export/merge_unsloth_lora.py \
    --base "$REF_MERGED" --adapter "checkpoints/$RUN_ID/adapter_final" \
    --out "checkpoints/${RUN_ID}_merged" --max-seq-length "$SEQ"
}

stage_eval(){
  local base_dir="checkpoints/$RUN_ID/eval_sft_baseline"
  local grpo_dir="checkpoints/$RUN_ID/eval_val"
  log "[eval] $KIND：SFT 基线 vs GRPO（同一 val / gen-batch 8 / 同分辨率）"

  if [[ "$KIND" == "q1" ]]; then
    uv run python -u eval/run_merged_eval.py --model "$REF_MERGED" --data "$VAL_Q1" \
      --out-dir "$base_dir" --prefix val --gen-batch-size 8 --max-pixels "$PX" --min-pixels "$MINPX"
    uv run python -u eval/run_unsloth_adapter_eval.py --model "$REF_MERGED" \
      --adapter "checkpoints/$RUN_ID/adapter_final" --data "$VAL_Q1" \
      --out-dir "$grpo_dir" --prefix val --gen-batch-size 8 --max-pixels "$PX" --min-pixels "$MINPX"
    uv run python -u - "$base_dir" "$grpo_dir" <<'PY'
import json, os, sys
def load(d):
    for n in ("val_report.json", "final_report.json"):
        p = os.path.join(d, "metrics", n)
        if os.path.isfile(p): return json.load(open(p, encoding="utf-8"))
    return {}
def g(r, k):
    return (r.get("overall") or r).get(k)
a, b = load(sys.argv[1]), load(sys.argv[2])
if not a or not b: print("(报告缺失)", sys.argv[1:]); raise SystemExit(1)
print(f"{'metric':24s} {'SFT':>8s} {'GRPO':>8s} {'Δ':>8s}")
for k in ("block_hit_rate", "empty_on_chrome_rate", "over_extraction_rate", "mean_edit_similarity", "format_leak_rate"):
    x, y = g(a, k), g(b, k)
    if x is not None and y is not None: print(f"{k:24s} {x:8.4f} {y:8.4f} {y-x:+8.4f}")
print("\n按桶 block_hit（门禁：multi_frag / semantic_group / real_labeled 各 +2pp）")
for bk in ("core_inner", "multi_frag", "semantic_group", "real_labeled"):
    x = (a.get("by_doc_type") or {}).get(bk) or {}
    y = (b.get("by_doc_type") or {}).get(bk) or {}
    if x and y:
        print(f"  {bk:16s} SFT={x.get('block_hit_rate',0):.4f} GRPO={y.get('block_hit_rate',0):.4f} "
              f"Δ={y.get('block_hit_rate',0)-x.get('block_hit_rate',0):+.4f} (n={x.get('n')})")
PY
  else
    uv run python -u eval/run_merged_eval.py --model "$REF_MERGED" --data "$VAL_Q2" \
      --out-dir "$base_dir" --prefix val --gen-batch-size 8 --max-pixels "$PX" --min-pixels "$MINPX"
    uv run python -u eval/run_unsloth_adapter_eval.py --model "$REF_MERGED" \
      --adapter "checkpoints/$RUN_ID/adapter_final" --data "$VAL_Q2" \
      --out-dir "$grpo_dir" --prefix val --gen-batch-size 8 --max-pixels "$PX" --min-pixels "$MINPX"
    for d in "$base_dir" "$grpo_dir"; do
      uv run python -u eval/run_q2_eval.py \
        --predictions "$d/metrics/final_predictions.jsonl" --split "$VAL_Q2" \
        --out-dir "$d/metrics" --prefix q2 >/dev/null
    done
    uv run python -u - "$base_dir" "$grpo_dir" <<'PY'
import json, os, sys
def load(d):
    p = os.path.join(d, "metrics", "q2_report.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
def g(r, k):
    return (r.get("overall") or r).get(k)
a, b = load(sys.argv[1]), load(sys.argv[2])
if not a or not b: print("(报告缺失)", sys.argv[1:]); raise SystemExit(1)
print(f"{'metric':28s} {'SFT':>8s} {'GRPO':>8s} {'Δ':>8s}")
for k in ("source_hit_rate", "translation_chrf", "xml_well_formed_rate",
          "effective_empty_rate", "hallucination_rate", "both_hit_rate", "format_leak_rate"):
    x, y = g(a, k), g(b, k)
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        print(f"{k:28s} {x:8.4f} {y:8.4f} {y-x:+8.4f}")
print("\n按桶 source_hit（门禁：multi_frag / semantic_group / real_labeled 各 +2pp）")
for bk in ("core_inner", "multi_frag", "semantic_group", "real_labeled"):
    x = ((a.get("by_bucket") or a.get("by_doc_type") or {}).get(bk) or {})
    y = ((b.get("by_bucket") or b.get("by_doc_type") or {}).get(bk) or {})
    if x and y:
        xa, yb = x.get("source_hit_rate", 0), y.get("source_hit_rate", 0)
        print(f"  {bk:16s} SFT={xa:.4f} GRPO={yb:.4f} Δ={yb-xa:+.4f} (n={x.get('n')})")
PY
  fi
}

case "${1:-}" in
  status) stage_status ;;
  cand) stage_cand ;;
  seed) stage_seed ;;
  probe) stage_probe ;;
  compose) stage_compose ;;
  dryrun) stage_dryrun ;;
  train) stage_train ;;
  merge) stage_merge ;;
  eval) stage_eval ;;
  all)
    [[ "${CONFIRM:-0}" == "1" ]] || { echo "长跑需要 CONFIRM=1"; exit 1; }
    stage_status; stage_cand; stage_probe; stage_compose; stage_dryrun; stage_train; stage_merge; stage_eval
    ;;
  *) sed -n '2,24p' "$0"; exit 1 ;;
esac

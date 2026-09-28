#!/usr/bin/env bash
# =============================================================================
# 0.8B 两轮 SFT 全流程（自包含 · 可重入 · 每步打印）
#
#   bash scripts/run_08b_full.sh          # 前台跑，直接看输出
#   SKIP_SMOKE=1 bash scripts/run_08b_full.sh
#
# 自包含：不依赖仓库里其它脚本（路径重定位用内联 python 完成）。
# 可重入：adapter_final / *_merged 存在就跳过；训练从最新 checkpoint-* 续。
# =============================================================================
set -Eeuo pipefail
trap 'echo "[$(date +%T)] !!! ERROR at line $LINENO: $BASH_COMMAND"' ERR

# ---------- 定位项目根（脚本在哪，项目根就在哪）----------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export UNSLOTH_CE_LOSS_N_CHUNKS=32
export TOKENIZERS_PARALLELISM=false
# 必须：Unsloth 会同时探测 AutoConfig 和 PeftConfig；后者对「没有 adapter_config.json
# 的本地目录」会当成 Hub repo id 去下载 adapter_config.json → 无网机器上 Errno 101 重试。
# 离线模式下该探测立即失败并被捕获，直接走本地全模型。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# hf-mirror 不代理 Xet 的 CAS（xethub.hf.co），不禁用会 401（仅在下模型时需要）
export HF_HUB_DISABLE_XET=1

log(){ echo; echo "==================== [$(date '+%F %T')] $* ===================="; }
gpu(){ nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"; }
# 注意：不能用 `ls -d glob`（无匹配返回非 0，配合 set -o pipefail 会直接触发 set -e）。
latest_ckpt(){
  [[ -d "$1" ]] || return 0
  find "$1" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1
}
q_done(){ [[ -f "checkpoints/$1/adapter_final/adapter_config.json" ]]; }

# ---------------------------- 参数 -------------------------------------------
MODEL="models/Qwen3.5-0.8B"
RUN_Q1="q1_08b"; RUN_Q2="q2_08b"
MERGED_Q1="checkpoints/${RUN_Q1}_merged"; MERGED_Q2="checkpoints/${RUN_Q2}_merged"
SEQ=12288; PX=8294400; MINPX=200704     # 2880² 保真，视觉 ≈8100 tok
B=4; GA=4                                # 有效 batch 16
Q1_TRAIN=data/splits_stage_q1_withreal/train.jsonl
Q1_VAL=data/splits_stage_q1_withreal/val.jsonl
Q1_TEST=data/splits_stage_q1_withreal/test.jsonl
Q2_TRAIN=data/splits_stage_q2_ocr_mt_v2/train.jsonl
Q2_VAL=data/splits_stage_q2_ocr_mt_v2/val.jsonl

log "[0/9] 环境与配置"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "uv           = $(command -v uv || echo '未找到！')"
echo "MODEL=$MODEL   RUN=$RUN_Q1 / $RUN_Q2   SEQ=$SEQ   PX=$PX   batch=${B}x${GA}"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE  TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "GPU: $(gpu)"
df -h "$PROJECT_ROOT" | tail -1
command -v uv >/dev/null 2>&1 || {
  echo "!! 找不到 uv。先跑："
  echo "   python3 -m pip install -U uv -i https://mirrors.aliyun.com/pypi/simple"
  echo "   export PATH=\$HOME/.local/bin:\$PATH"
  exit 1
}
if pgrep -f "unsloth_stage_a.py" >/dev/null 2>&1; then
  echo "!! 已有训练进程在跑，先停： pkill -f unsloth_stage_a.py"; exit 1
fi

log "[1/9] 修正 split 里的图片绝对路径（就地重写到当前项目根）"
uv run python -u - <<'PY'
import glob, json, os
root = os.getcwd()
files = [f for d in sorted(glob.glob("data/splits_*"))
         for f in sorted(glob.glob(os.path.join(d, "*.jsonl")))]
if not files:
    raise SystemExit("找不到 data/splits_*/*.jsonl —— 项目根不对？")
example = None
tot_files = tot_changed = tot_rows = 0
for f in files:
    lines = open(f, encoding="utf-8").read().splitlines()
    out, ch = [], 0
    for ln in lines:
        if not ln.strip():
            out.append(ln); continue
        row = json.loads(ln)
        imgs = row.get("images") or []
        raw = imgs[0] if imgs else ""
        # 绝对路径 + 已失效 + 含 /data/ → 重挂到 <项目根>/data/...
        if raw.startswith("/") and "/data/" in raw and not os.path.exists(raw):
            cand = os.path.join(root, "data", raw.split("/data/", 1)[1])
            if os.path.exists(cand):
                row["images"] = [cand] + list(imgs[1:]); ch += 1
                if example is None:
                    example = f"{raw}\n      -> {cand}"
        out.append(json.dumps(row, ensure_ascii=False))
    tot_rows += len(lines)
    if ch:
        tot_files += 1; tot_changed += ch
        open(f, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"  {f}: 改写 {ch}/{len(lines)} 行")
print(f"[relocate] {tot_files} 个文件 / {tot_changed} 行（共扫描 {tot_rows} 行）")
if example:
    print("  示例:\n    " + example)
PY

log "[2/9] 自检：模型 + 数据"
uv run python -u - <<'PY'
import json, os, sys
m = "models/Qwen3.5-0.8B"
ok = os.path.isdir(m) and os.path.isfile(f"{m}/config.json") and any(
    f.startswith("model.safetensors") for f in os.listdir(m))
print(("OK  " if ok else "FAIL"), m)
if not ok:
    sys.exit("模型不完整： uv run hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B")
for s in ["data/splits_stage_q1_withreal/train.jsonl",
          "data/splits_stage_q1_withreal/val.jsonl",
          "data/splits_stage_q2_ocr_mt_v2/train.jsonl",
          "data/splits_stage_q2_ocr_mt_v2/val.jsonl"]:
    n = miss = 0
    for ln in open(s, encoding="utf-8"):
        if not ln.strip():
            continue
        n += 1
        imgs = json.loads(ln).get("images") or []
        if imgs and not os.path.exists(imgs[0]):
            miss += 1
    print(f"  {s}: rows={n} missing_images={miss}")
    if miss:
        sys.exit(f"缺图 {miss} 张：images.tar 没解压完整，或重定位未生效")
print("data OK")
PY

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
log "[3/9] 冒烟（8 samples / 2 steps）"
rm -rf /tmp/smoke_08b
uv run python -u train/unsloth_stage_a.py \
  --data "$Q1_TRAIN" --val "$Q1_VAL" --model "$MODEL" --out /tmp/smoke_08b \
  --max-seq-length $SEQ --batch-size $B --grad-accum $GA \
  --lora-rank 32 --lora-alpha 64 --finetune-vision \
  --max-samples 8 --max-steps 2 --skip-post-eval --num-workers 0 --no-pin-memory \
  --max-pixels $PX --min-pixels $MINPX --collator-resize max
echo "冒烟通过。GPU: $(gpu)"
else
log "[3/9] 冒烟已跳过（SKIP_SMOKE=1）"
fi

if q_done "$RUN_Q1"; then
  log "[4/9] SFT_Q1 已完成，跳过"
else
log "[4/9] SFT_Q1 开始（2397 步）"
CK="$(latest_ckpt "checkpoints/$RUN_Q1")"; RES=()
[[ -n "$CK" ]] && { echo "从 $CK 续训"; RES=(--resume-from-checkpoint "$CK"); }
uv run python -u train/unsloth_stage_a.py \
  --data "$Q1_TRAIN" --val "$Q1_VAL" --model "$MODEL" \
  --out-root checkpoints --run-id "$RUN_Q1" \
  --max-seq-length $SEQ --batch-size $B --grad-accum $GA \
  --lr 5e-5 --epochs 1.5 --lora-rank 32 --lora-alpha 64 --finetune-vision \
  --save-steps 250 --eval-steps 250 --logging-steps 10 --eval-batch-size 2 \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4 --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --max-pixels $PX --min-pixels $MINPX --collator-resize max \
  --post-eval-batch-size 4 --seed 42 ${RES[@]+"${RES[@]}"}
fi

if [[ -f "$MERGED_Q1/config.json" ]]; then
  log "[5/9] merge Q1 已存在，跳过"
else
log "[5/9] merge Q1 -> $MERGED_Q1"
uv run python -u export/merge_unsloth_lora.py --base "$MODEL" \
  --adapter "checkpoints/$RUN_Q1/adapter_final" --out "$MERGED_Q1" --max-seq-length $SEQ
fi

if q_done "$RUN_Q2"; then
  log "[6/9] SFT_Q2 已完成，跳过"
else
log "[6/9] SFT_Q2 开始（3101 步，冻结 vision）"
CK="$(latest_ckpt "checkpoints/$RUN_Q2")"; RES=()
[[ -n "$CK" ]] && { echo "从 $CK 续训"; RES=(--resume-from-checkpoint "$CK"); }
uv run python -u train/unsloth_stage_a.py \
  --data "$Q2_TRAIN" --val "$Q2_VAL" --model "$MERGED_Q1" \
  --out-root checkpoints --run-id "$RUN_Q2" \
  --max-seq-length $SEQ --batch-size $B --grad-accum $GA \
  --lr 1.5e-5 --epochs ${Q2_EPOCHS:-1.0} --lora-rank 32 --lora-alpha 64 \
  --save-steps 250 --eval-steps 250 --logging-steps 10 --eval-batch-size 1 \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4 --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --max-pixels $PX --min-pixels $MINPX --collator-resize max \
  --post-eval-batch-size 2 --seed 42 ${RES[@]+"${RES[@]}"}
fi

if [[ -f "$MERGED_Q2/config.json" ]]; then
  log "[7/9] merge Q2 已存在，跳过"
else
log "[7/9] merge Q2 -> $MERGED_Q2"
uv run python -u export/merge_unsloth_lora.py --base "$MERGED_Q1" \
  --adapter "checkpoints/$RUN_Q2/adapter_final" --out "$MERGED_Q2" --max-seq-length $SEQ
fi

log "[8/9] Q1 test 评测"
uv run python -u eval/run_unsloth_adapter_eval.py \
  --model "$MODEL" --adapter "checkpoints/$RUN_Q1/adapter_final" \
  --data "$Q1_TEST" --gen-batch-size 8 --max-pixels $PX --min-pixels $MINPX \
  --out-dir "checkpoints/$RUN_Q1/eval_test" --prefix test

log "[9/9] Q2 拆分指标 + 汇总"
uv run python -u eval/run_q2_eval.py \
  --predictions "checkpoints/$RUN_Q2/metrics/final_predictions.jsonl" \
  --split "$Q2_VAL" --out-dir "checkpoints/$RUN_Q2/metrics" --prefix q2

echo; echo "################ 全部完成 ################"
echo "--- SFT_Q1（val，post-eval 自动产出）---"; cat "checkpoints/$RUN_Q1/metrics/final_report.json"; echo
echo "--- SFT_Q2 报告 ---"; cat "checkpoints/$RUN_Q2/metrics/q2_report.md"
echo "--- 可部署整模 ---"; ls -lh "$MERGED_Q2"

# 32GB 卡：训练 0.8B 与 4B（完整启动命令）

日期：2026-09-28。目标：在单张 **32GB** 卡上走完 0.8B / 4B 的两轮 SFT（Q1 → merge → Q2 → merge → 评测）。
与本地 8GB 的差别：**用满 2880² 保真 + 放大 micro-batch + 不再需要 WSL 内存 workaround**。

前置（见 [`CLOUD_PACKAGING.md`](CLOUD_PACKAGING.md)）：`scripts/bootstrap_cloud_env.sh` → 下模型 →
`scripts/relocate_paths.py --apply` → `uv run pytest -q`。

---

## 0. 推荐超参总表

| | **0.8B Q1** | **0.8B Q2** | **4B Q1** | **4B Q2** |
|---|---|---|---|---|
| 底模 | `models/Qwen3.5-0.8B` | `checkpoints/q1_08b_merged` | `models/Qwen3.5-4B` | `checkpoints/q1_4b_merged` |
| `--max-seq-length` | **12288** | 12288 | **12288** | 12288 |
| `--max-pixels` | **8294400**（2880²，视觉≈8100 tok） | 8294400 | 8294400 | 8294400 |
| `--batch-size` × `--grad-accum` | **4 × 4**（有效 16） | 4 × 4 | **2 × 8** | 2 × 8 |
| `--lr` | 5e-5 | 1.5e-5 | **3e-5** | **1e-5** |
| `--epochs` | 1.5 | 1.0 | **1.0**（论文 4B 只训 0.2ep，见 §5） | 1.0 |
| LoRA | r32 α64 | r32 α64 | r32 α64 | r32 α64 |
| vision | `--finetune-vision` | 冻结（不加 flag） | `--finetune-vision` | 冻结 |
| `--eval-batch-size` | 2 | 1 | 1 | 1 |
| `--post-eval-batch-size` | 4 | 2 | 2 | 2 |
| loader | `--num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4` | 同 | 同 | 同 |
| 预估显存峰值 | ~10 GB | ~9 GB | ~10 GB | ~9 GB |

> 依据（本地 8GB 实测，batch 1–2，最密页）：0.8B 2880²/12288 → 5.8GB；2B → 6.7GB；4B bs1 → 6.6GB。
> 32GB 上按上表放大 micro-batch 后仍在 ~10GB 量级，留足余量给 eval / 碎片。
> **改动前请先跑 §5 的冒烟**，照实测微调。

`--min-pixels 200704`、`--collator-resize max`、`--save-steps 250 --eval-steps 250 --logging-steps 10`、
`--warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine --optim adamw_8bit --max-grad-norm 1.0 --seed 42`
四组命令里都一样，下文命令已完整写出。

---

## 1. 0.8B

### 1.1 SFT_Q1
```bash
cd ~/autodl-tmp/ocr-mt-finetuning
source scripts/setup_autodl_mirrors.sh

uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q1_withreal/train.jsonl \
  --val  data/splits_stage_q1_withreal/val.jsonl \
  --model models/Qwen3.5-0.8B \
  --out-root checkpoints --run-id q1_08b \
  --max-seq-length 12288 \
  --batch-size 4 --grad-accum 4 \
  --lr 5e-5 --epochs 1.5 \
  --lora-rank 32 --lora-alpha 64 --finetune-vision \
  --save-steps 250 --eval-steps 250 --logging-steps 10 --eval-batch-size 2 \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4 --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --max-pixels 8294400 --min-pixels 200704 --collator-resize max \
  --post-eval-batch-size 4 --seed 42
```
（断线续训：追加 `--resume-from-checkpoint checkpoints/q1_08b/checkpoint-<N>`，其余参数保持一致。）

### 1.2 merge Q1
```bash
uv run python export/merge_unsloth_lora.py \
  --base models/Qwen3.5-0.8B \
  --adapter checkpoints/q1_08b/adapter_final \
  --out checkpoints/q1_08b_merged \
  --max-seq-length 12288
```

### 1.3 SFT_Q2（冻结 vision）
```bash
uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q2_ocr_mt_v2/train.jsonl \
  --val  data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --model checkpoints/q1_08b_merged \
  --out-root checkpoints --run-id q2_08b \
  --max-seq-length 12288 \
  --batch-size 4 --grad-accum 4 \
  --lr 1.5e-5 --epochs 1.0 \
  --lora-rank 32 --lora-alpha 64 \
  --save-steps 250 --eval-steps 250 --logging-steps 10 --eval-batch-size 1 \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4 --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --max-pixels 8294400 --min-pixels 200704 --collator-resize max \
  --post-eval-batch-size 2 --seed 42
```

### 1.4 merge Q2 + 评测
```bash
uv run python export/merge_unsloth_lora.py \
  --base checkpoints/q1_08b_merged \
  --adapter checkpoints/q2_08b/adapter_final \
  --out checkpoints/q2_08b_merged --max-seq-length 12288

# Q1 test（可选）
uv run python eval/run_unsloth_adapter_eval.py \
  --model models/Qwen3.5-0.8B --adapter checkpoints/q1_08b/adapter_final \
  --data data/splits_stage_q1_withreal/test.jsonl \
  --gen-batch-size 8 --max-pixels 8294400 --min-pixels 200704 \
  --out-dir checkpoints/q1_08b/eval_test --prefix test

# Q2 拆分指标（跑在 Q2 SFT 自动 post-eval 的 predictions 上；无 GPU）
uv run python eval/run_q2_eval.py \
  --predictions checkpoints/q2_08b/metrics/final_predictions.jsonl \
  --split data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --out-dir checkpoints/q2_08b/metrics --prefix q2
```

### 1.5 一键等价命令
```bash
CONFIRM=1 MODEL=models/Qwen3.5-0.8B RUN_Q1=q1_08b RUN_Q2=q2_08b \
  MAX_PIXELS=8294400 MAX_SEQ_LENGTH=12288 BATCH_SIZE=4 GRAD_ACCUM=4 \
  NUM_WORKERS=8 PIN_MEMORY=1 PERSISTENT_WORKERS=1 PREFETCH=4 \
  EVAL_BATCH_Q1=2 EVAL_BATCH_Q2=1 POST_EVAL_BATCH_Q1=4 POST_EVAL_BATCH_Q2=2 \
  bash train/run_08b_pipeline.sh all
```

---

## 2. 4B

### 2.1 SFT_Q1
```bash
uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q1_withreal/train.jsonl \
  --val  data/splits_stage_q1_withreal/val.jsonl \
  --model models/Qwen3.5-4B \
  --out-root checkpoints --run-id q1_4b \
  --max-seq-length 12288 \
  --batch-size 2 --grad-accum 8 \
  --lr 3e-5 --epochs 1.0 \
  --lora-rank 32 --lora-alpha 64 --finetune-vision \
  --save-steps 250 --eval-steps 250 --logging-steps 10 --eval-batch-size 1 \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4 --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --max-pixels 8294400 --min-pixels 200704 --collator-resize max \
  --post-eval-batch-size 2 --seed 42
```

### 2.2 merge Q1
```bash
uv run python export/merge_unsloth_lora.py \
  --base models/Qwen3.5-4B \
  --adapter checkpoints/q1_4b/adapter_final \
  --out checkpoints/q1_4b_merged --max-seq-length 12288
```

### 2.3 SFT_Q2（冻结 vision）
```bash
uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q2_ocr_mt_v2/train.jsonl \
  --val  data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --model checkpoints/q1_4b_merged \
  --out-root checkpoints --run-id q2_4b \
  --max-seq-length 12288 \
  --batch-size 2 --grad-accum 8 \
  --lr 1e-5 --epochs 1.0 \
  --lora-rank 32 --lora-alpha 64 \
  --save-steps 250 --eval-steps 250 --logging-steps 10 --eval-batch-size 1 \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4 --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --max-pixels 8294400 --min-pixels 200704 --collator-resize max \
  --post-eval-batch-size 2 --seed 42
```

### 2.4 merge Q2 + 评测
```bash
uv run python export/merge_unsloth_lora.py \
  --base checkpoints/q1_4b_merged \
  --adapter checkpoints/q2_4b/adapter_final \
  --out checkpoints/q2_4b_merged --max-seq-length 12288

uv run python eval/run_unsloth_adapter_eval.py \
  --model models/Qwen3.5-4B --adapter checkpoints/q1_4b/adapter_final \
  --data data/splits_stage_q1_withreal/test.jsonl \
  --gen-batch-size 4 --max-pixels 8294400 --min-pixels 200704 \
  --out-dir checkpoints/q1_4b/eval_test --prefix test

uv run python eval/run_q2_eval.py \
  --predictions checkpoints/q2_4b/metrics/final_predictions.jsonl \
  --split data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --out-dir checkpoints/q2_4b/metrics --prefix q2
```

### 2.5 一键等价命令
```bash
CONFIRM=1 MODEL=models/Qwen3.5-4B RUN_Q1=q1_4b RUN_Q2=q2_4b \
  MAX_PIXELS=8294400 MAX_SEQ_LENGTH=12288 BATCH_SIZE=2 GRAD_ACCUM=8 \
  NUM_WORKERS=8 PIN_MEMORY=1 PERSISTENT_WORKERS=1 PREFETCH=4 \
  LR_Q1=3e-5 LR_Q2=1e-5 EPOCHS_Q1=1.0 EPOCHS_Q2=1.0 \
  EVAL_BATCH_Q1=1 EVAL_BATCH_Q2=1 POST_EVAL_BATCH_Q1=2 POST_EVAL_BATCH_Q2=2 \
  bash train/run_08b_pipeline.sh all
```

---

## 3. 长跑：后台 + 日志 + 自愈
```bash
mkdir -p logs
setsid nohup env CONFIRM=1 MODEL=models/Qwen3.5-4B RUN_Q1=q1_4b RUN_Q2=q2_4b \
  MAX_PIXELS=8294400 MAX_SEQ_LENGTH=12288 BATCH_SIZE=2 GRAD_ACCUM=8 \
  NUM_WORKERS=8 PIN_MEMORY=1 PERSISTENT_WORKERS=1 \
  LR_Q1=3e-5 LR_Q2=1e-5 EPOCHS_Q1=1.0 EPOCHS_Q2=1.0 \
  bash train/run_08b_pipeline.sh all > logs/q4b.log 2>&1 < /dev/null &

# 自愈（挂了自动从最新 checkpoint 续）+ 资源看门狗
setsid nohup env MODEL=models/Qwen3.5-4B RUN_Q1=q1_4b RUN_Q2=q2_4b NUM_WORKERS=8 \
  bash scripts/supervise_08b.sh > logs/q4b_supervise.log 2>&1 < /dev/null &
setsid nohup env WATCH_PAT=supervise_08b.sh KILL_PATS="run_08b_pipeline.sh unsloth_stage_a.py" \
  bash scripts/watch_08b_training.sh > logs/q4b_watchdog.log 2>&1 < /dev/null &
```
（32GB 卡的宿主内存通常很大，看门狗的内存阈值可按需放；它默认只在 `MemAvailable<800MB×3` 时动手。）

---

## 4. 开跑前必做：冒烟（约 3–5 分钟）
```bash
# 0.8B
CONFIRM=1 MODEL=models/Qwen3.5-0.8B MAX_PIXELS=8294400 MAX_SEQ_LENGTH=12288 \
  BATCH_SIZE=4 GRAD_ACCUM=4 bash train/run_08b_pipeline.sh smoke
# 4B
CONFIRM=1 MODEL=models/Qwen3.5-4B MAX_PIXELS=8294400 MAX_SEQ_LENGTH=12288 \
  BATCH_SIZE=2 GRAD_ACCUM=8 bash train/run_08b_pipeline.sh smoke
```
看日志里的两行：`~8100 vision tokens/image`、`collator_resize=max`，并盯 `nvidia-smi` 峰值。

**OOM 降级顺序**：`--batch-size` 减半、`--grad-accum` 加倍（保持有效 batch）→ `--eval-batch-size 1` →
`--max-pixels 6291456`（≈2500²，视觉 ~6100）→ `--max-pixels 4194304` + `--max-seq-length 8192`。
**不要**降 seq 而不降 pixels（会截掉答案）。

---

## 5. 可选第二阶段：GRPO / OPD（32GB 才做得动）

**4B GRPO（论文正路：只在 4B 分支做 RL）**
```bash
# 1) 探针：G=8 rollout，筛“有学习价值”的样本
uv run python eval/run_grpo_value_probe.py \
  --src data/splits_stage_q1_withreal/train.jsonl \
  --model models/Qwen3.5-4B --adapter checkpoints/q1_4b/adapter_final \
  --out checkpoints/grpo_value_probe_4b \
  --reward-set q1v2 --scenes regular,multi_frag,semantic_group --include-empty \
  --g 8 --gen-batch 2 --temperature 1.2 --top-p 1.0 --max-new-tokens 256 \
  --max-seq-length 12288 --max-pixels 8294400

# 2) 组数据
uv run python data/scripts/compose_grpo_q1_v2.py

# 3) GRPO（--adapter + 不加 --continue-adapter = 烘成冻结 π_ref + 新挂 r16 LoRA）
uv run python train/unsloth_grpo.py \
  --data data/splits_grpo_q1_v2/train.jsonl --val data/splits_grpo_q1_v2/val.jsonl \
  --model models/Qwen3.5-4B --adapter checkpoints/q1_4b/adapter_final \
  --run-id grpo_q1_4b --reward-set q1v2 \
  --num-generations 8 --temperature 1.0 --top-p 1.0 --beta 0.04 \
  --max-completion-length 256 --gen-chunk-size 2 \
  --batch-size 4 --grad-accum 4 --lr 5e-6 --epochs 1.0 --lora-rank 16 --lora-alpha 32 \
  --save-steps 50 --logging-steps 5 \
  --max-seq-length 12288 --max-pixels 8294400 --num-workers 8 \
  --max-grad-norm 1.0 --optim adamw_8bit --seed 42
```
> 注意：Q1 SFT 后 signal 样本可能很少（本地 2B 实测只有 25%）。若探针产出 <~300 条有学习价值的样本，
> GRPO 收益有限——这也是论文把 RL 放在 4B 分支的原因。

**OPD（4B → 0.8B）**：⚠️ **本仓库还没有实现**。论文用的是「学生在自己轨迹上生成 → 教师在学生 top-k 支撑集上给
log-probs → 学生做 top-k reverse-KL」，需要 token 级 teacher logits + 自研 trainer（Unsloth/TRL 不支持）。
32GB 卡足够跑，但需要先写这部分代码；要做的话我可以单独实现（含 test）。

---

## 6. 监控
```bash
tail -f logs/q4b.log
tail -f logs/q4b_supervise.log
tail -f logs/q4b_watchdog.log
watch -n 5 'nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader'
# TensorBoard
uv run tensorboard --logdir checkpoints/q1_4b/tb --port 6006 --bind_all
```

## 7. 与本地 8GB 的差异（别忘了）
- `--max-seq-length 12288`（2880² 必需）；本地旧值是 8192。
- 不再需要 `--no-pin-memory` / `--num-workers 0`（那是 WSL 15GB 的 workaround）。
- 训练/评测/Studio 的 `--max-pixels` 必须一致（都是 8294400）；Studio：`MAX_PIXELS=8294400 bash eval/marker_studio/run_q2.sh`。

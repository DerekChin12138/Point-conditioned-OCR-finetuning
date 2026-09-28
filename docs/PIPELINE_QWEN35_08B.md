# Qwen3.5-0.8B 两轮 SFT —— 对齐 2B 口径的干净基线

日期：2026-09-26。目标：**消费级终端部署**，先用与 2B **完全相同的配方**重打 0.8B 基线，
从而把「容量差异」从「数据/协议差异」里分离出来。

> ⚠️ **本文档只准备脚本；训练需显式执行 `CONFIRM=1`。** 不要自动开训。

---

## 0. 为什么必须重跑：旧 0.8B 数字被至少 5 个因素污染

| 混淆 | 旧 0.8B（`q1_withreal` / `q2_ocr_mt`） | 新口径（与 2B 一致） |
|---|---|---|
| 序列长度 | **4096**（当时以为会截断） | **8192** |
| **有效分辨率** | **768px 宽**（collator 静默裁剪；`--max-pixels 8294400` 未生效） | **2048px 宽**（`--collator-resize max`，`--max-pixels 2048²`） |
| 准星面积 | 旧 `x45c/x45d`，`_archive_v1` 实测 mean **0.00335** | `x45r` mean **0.0014**（0.08%–0.2%） |
| 负例数据 | 未清洗（625 负例，prose 表格 / clear-on-ink） | `splits_stage_q2_ocr_mt_v2`（已清洗 + blank×1.35） |
| Q2 底模 | **`q1_grpo_merged`（已证伪的 GRPO）** | 纯 SFT_Q1 整模 |
| `--finetune-vision` | **no-op bug**（Q2 的 vision 实际被训练） | 已修：Q1 训 vision、Q2 冻 vision |
| 精度 | Q1 是 bf16（`no_4bit`） | 默认 4bit（与 2B 等效比较） |

→ 旧 `block_hit 87.6%` / `source_hit 89.3%` / `chrF 0.341` **不能代表 0.8B 上限**。

> ⚠️ **2026-09-28 重大更正**：上表“序列长度”这一列的前提是错的——训练时
> `UnslothVisionDataCollator(resize="min")` 把图像宽度静默裁到 **768**，`--max-pixels 8294400` 根本没生效
> （实测最密页只有 ~336 视觉 token，序列 ~350–810，4096 完全够）。
> 而**评测/Studio 走 processor 没有这个上限（全分辨率）→ train/eval 分辨率不一致**。
> 已修复：`--collator-resize`（默认 `max`）+ 分辨率由 `--max-pixels` 控制。
> 本目录 2026-09-28 之前的所有结果（包括 `checkpoints/q{1,2}_08b`）都是 **768px 训练**的产物，需重跑。
> 详见 [`README.md`](../README.md) §六 “2026-09-28 修复”。

---

## 1. 一键脚本

```bash
# 先验证脚本与显存（8 samples / 2 steps，不写正式 run）
bash train/run_08b_pipeline.sh smoke

# 正式跑（每步都要 CONFIRM=1）
CONFIRM=1 bash train/run_08b_pipeline.sh q1        # SFT_Q1  → checkpoints/q1_08b/
bash train/run_08b_pipeline.sh merge-q1            #        → checkpoints/q1_08b_merged/
CONFIRM=1 bash train/run_08b_pipeline.sh q2        # SFT_Q2  → checkpoints/q2_08b/
bash train/run_08b_pipeline.sh merge-q2            #        → checkpoints/q2_08b_merged/（部署用）
bash train/run_08b_pipeline.sh eval-q1             # Q1 test 评测
bash train/run_08b_pipeline.sh eval-q2             # Q2 val 拆分指标

# 或者串起来（长跑，~9–14h）
CONFIRM=1 bash train/run_08b_pipeline.sh all
```

环境变量覆盖（都有默认值）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL` | `models/Qwen3.5-0.8B` | 基座 |
| `PRECISION` | `4bit` | `4bit`（对齐 2B）或 `bf16`（0.8B 显存够，质量略好，但非等效比较） |
| `NUM_WORKERS` | `0` | 宿主 15GB 很紧；>0 会 fork COW 复制 |
| `RUN_Q1` / `RUN_Q2` | `q1_08b` / `q2_08b` | 产出目录名 |
| `Q1_TRAIN/VAL/TEST`、`Q2_TRAIN/VAL` | 见脚本头 | 数据路径 |

---

## 2. 手工等价命令（若要复现/微调）

```bash
# SFT_Q1
uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q1_withreal/train.jsonl \
  --val  data/splits_stage_q1_withreal/val.jsonl \
  --model models/Qwen3.5-0.8B \
  --out-root checkpoints --run-id q1_08b \
  --max-seq-length 8192 --batch-size 2 --grad-accum 8 \
  --collator-resize max \
  --lr 5e-5 --epochs 1.5 --lora-rank 32 --lora-alpha 64 --finetune-vision \
  --save-steps 500 --eval-steps 250 --logging-steps 10 --eval-batch-size 2 \
  --num-workers 0 --no-pin-memory --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --max-pixels 4194304 --min-pixels 200704 \
  --post-eval-batch-size 8 --seed 42

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python export/merge_unsloth_lora.py \
  --base models/Qwen3.5-0.8B --adapter checkpoints/q1_08b/adapter_final \
  --out checkpoints/q1_08b_merged --max-seq-length 8192

# SFT_Q2（底模 = Q1 整模，冻结 vision）
uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q2_ocr_mt_v2/train.jsonl \
  --val  data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --model checkpoints/q1_08b_merged \
  --out-root checkpoints --run-id q2_08b \
  --max-seq-length 8192 --batch-size 2 --grad-accum 8 \
  --collator-resize max \
  --lr 1.5e-5 --epochs 1.0 --lora-rank 32 --lora-alpha 64 \
  --save-steps 500 --eval-steps 250 --logging-steps 10 --eval-batch-size 1 \
  --num-workers 0 --no-pin-memory --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --max-pixels 4194304 --min-pixels 200704 \
  --post-eval-batch-size 2 --seed 42

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python export/merge_unsloth_lora.py \
  --base checkpoints/q1_08b_merged --adapter checkpoints/q2_08b/adapter_final \
  --out checkpoints/q2_08b_merged --max-seq-length 8192

# 评测
uv run python eval/run_q2_eval.py \
  --predictions checkpoints/q2_08b/metrics/final_predictions.jsonl \
  --split data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --out-dir checkpoints/q2_08b/metrics --prefix q2
```

冒烟实测（4bit, 8 samples）：2 步 ~35s（step1 含编译），GPU 远低于 8GB，宿主 RAM 稳定。

---

## 3. 预算

| 阶段 | 0.8B 实测/估算 | 对照 2B |
|---|---|---|
| 吞吐（稳态） | ~**0.4–0.55 s/sample** | 0.78 s/sample |
| SFT_Q1（1.5 ep） | ~**4–6h** | 8.4h |
| SFT_Q2（1.0 ep） | ~**5–7.5h** | 10.7h |
| 显存峰值 | **~2GB（768px 训练，已废弃）**；2048² 约 3.4GB reserved，2880² 约 5.8GB | 3.9GB（同样受 768 影响） |
| **合计** | **~9–14h** | 19.1h |

### DataLoader workers 标定（2026-09-26 实测，本机）

Q1 数据、同配方、6 步/96 样本：

| `--num-workers` | throughput | GPU peak | RAM peak used |
|---|---|---|---|
| 0 | 0.94 samples/s | 2.6GB | 6.25GB |
| 4 | 1.66 samples/s | 2.4GB | 6.74GB |
| **8** | **1.73 samples/s** | 2.2GB | 7.59GB |

→ 脚本默认已改为 **8**（正式长跑时 RAM 稳在 ~11.1GB used / ~4.6GB avail，不涨）。
宿主更小时用 `NUM_WORKERS=4`（损失 ~4% 吞吐，省 RAM）。

---

## 4. 跑完后要做的对照（判定门禁）

| 指标 | 旧 0.8B（污染） | 0.8B 新（待测） | 2B（基准） | 结论含义 |
|---|---|---|---|---|
| Q1 `block_hit` | 0.876 | ? | **0.9396** | ≥0.90 → 0.8B 可部署；<0.88 → 需蒸馏 |
| Q1 `semantic_group` | 0.766 | ? | 0.971 | 容量对「组合语义」的敏感度 |
| Q1 `multi_frag` | 0.752 | ? | 0.856 | 同上 |
| Q2 `source_hit` | 0.893 | ? | 0.914 | 定位是否回退 |
| Q2 `effective_empty` | 0.658 | ? | 0.996 | 数据修复是否对 0.8B 同样有效 |
| Q2 `translation_chrf` | 0.341 | ? | 0.392 | 翻译轴的容量差距 |

**决策**：
- 若 0.8B 新基线 `block_hit ≥ 0.90` 且 `effective_empty ≥ 0.98` → 直接以 0.8B 为目标，后续只做翻译增强（两阶段 / 蒸馏）。
- 若 `< 0.88` → 说明 0.8B 定位确实不足，需要 2B/4B teacher 蒸馏或两阶段（0.8B 只做点 OCR）。
- 两种情况下都应先做 `docs/STRATEGY_REVIEW_4B_OPD.md` 的 **P-0（统一翻译参考 + 测延迟）**。

---

## 5. 后续（与本轮衔接）

1. **推理延迟基准**：0.8B vs 2B 在真实 2–9MP 截图上的 p50/p95 + 峰值内存（消费级终端的关键数字）。
2. **4B teacher**：本地 4B SFT 可行（~9h，0.2–0.5 ep，见 `eval/results/qwen4b_feasibility/`）；4B GRPO + OPD 上云。
3. **OPD**：需要自研 top-k reverse-KL trainer（Unsloth/TRL 无）；论文 OPD 的 teacher 是 **4B+RL**，不是 4B SFT。

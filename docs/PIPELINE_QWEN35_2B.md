# 完整开训流程（Qwen3.5-2B）

日期：2026-09-23。本文是**从零到可部署**的完整命令清单。
标记协议：`x45r`，**每图均匀随机 0.08%–0.2% 面积**（见 [`MARKER_SPEC.md`](MARKER_SPEC.md)）。
所有 `run-id` / 路径都是新目录，**不覆盖** 0.8B 的已有产物。

> 机器：RTX 4060 Laptop 8GB / i9-13980HX 32 线程 / 15GB RAM / `/dev/shm` 7.8GB。
> 实测：2B 4bit + 8192 序列 → 峰值 **3.9GB**，**~1.25 s/sample**。
>
> ⚠️ **2026-09-28 更正**：上面这个 3.9GB / “8.3MP” 实际是 **768px 宽**的结果——
> `UnslothVisionDataCollator(resize="min")` 把图像宽度裁到 `vision_config.image_size=768`，
> `--max-pixels 8294400` 并未生效（实测最密页只有 ~336 视觉 token）。因此：
> ① “最密页 seq_len ≈ 8000，不要用 4096” 这个理由**不成立**（那是固定尺寸下的期望值，实际只有 ~350–810）；
> ② 本文件的 2B 结果都是 **768px 训练 + 全分辨率评测**（train/eval 不一致）得到的，分辨率修复后需重跑基线。
> 新默认：`--collator-resize max`（不裁剪），分辨率由 `--max-pixels` 控制，视觉 token ≈ `max_pixels/1024`。

---

## 0. 环境

```bash
cd /home/derek_qxc/workspace/ocr-finetuning/Point-conditioned-OCR-finetuning
uv sync --extra dev --extra train
uv pip install unsloth
uv run pytest -q                       # 应 148 passed
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1    # 2B 已下到 models/Qwen3.5-2B
```

---

## 1. 重渲全部标记（新协议）

标记尺寸变了，**所有池必须重渲**（旧 jpg 不能复用）。

```bash
# 1.1 Q1 合成池（core_inner / empty_clear / empty_special / multi_frag / semantic_group）
WORKERS=12 bash data/scripts/build_q1_pools_sequential.sh

# 1.2 Q2 合成池（OCR+MT）
WORKERS=12 bash data/scripts/build_q2_pools_sequential.sh

# 1.3 真实截图池：重打标（随机尺寸）+ 克制补中文
uv run python data/scripts/build_real_label_pool.py
uv run python data/scripts/build_real_mt_pool.py
```

想调范围（默认 0.0015 / 0.003，即 0.15%–0.3%）：

```bash
uv run python data/scripts/build_pool.py --pool-id core_inner --out-root data/pools_q \
  --target 18000 --chrome --wipe --workers 12 --seed 42 --prompt-key a2_v3 \
  --marker-area-min 0.0008 --marker-area-max 0.002
```

---

## 2. 组 SFT 划分

```bash
# Q1（合成 + 真实）
uv run python data/scripts/compose_q1_withreal.py

# Q2（合成 + 真实，ocr_mt_v1）
uv run python data/scripts/compose_q2_ocr_mt.py

# Q2 负例清洗 → data/splits_stage_q2_ocr_mt_v2/
uv run python data/scripts/filter_q2_negatives.py --mode splits --blank-factor 1.35
```

快速自查（可选）：

```bash
uv run python - <<'PY'
import json, collections
for p in ["data/splits_stage_q1_withreal/train.jsonl",
          "data/splits_stage_q2_ocr_mt_v2/train.jsonl"]:
    c = collections.Counter(); fracs = []
    for l in open(p, encoding="utf-8"):
        m = json.loads(l)["metadata"]; c[m.get("pool_id")] += 1
        if m.get("marker_area_frac"): fracs.append(m["marker_area_frac"])
    print(p, dict(c), "marker_frac", round(min(fracs),4), "-", round(max(fracs),4))
PY
```

---

## 3. SFT_Q1（2B）

```bash
uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q1_withreal/train.jsonl \
  --val  data/splits_stage_q1_withreal/val.jsonl \
  --model models/Qwen3.5-2B \
  --out-root checkpoints --run-id q1_2b \
  --max-seq-length 8192 \
  --batch-size 2 --grad-accum 8 \
  --lr 5e-5 --epochs 1.5 \
  --lora-rank 32 --lora-alpha 64 \
  --finetune-vision \
  --save-steps 500 --eval-steps 250 --logging-steps 10 --eval-batch-size 2 \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4 --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --max-pixels 8294400 --min-pixels 200704 \
  --post-eval-batch-size 8 --seed 42
```

- `--finetune-vision` 现在**真的生效**（0.8B 时代是 no-op，见 [`Q2_PLAN.md`](Q2_PLAN.md) §2.6）。标记几何变了，vision 需要适应。若显存吃紧就去掉这个 flag（会冻结 vision LoRA，省 ~12.6M 可训练参数）。
- 期望：峰值 ~4–5GB；门禁 `block_hit ≥ 0.88`、`empty_on_chrome ≥ 0.95`。

---

## 4. 评 SFT_Q1 + merge

```bash
uv run python eval/run_unsloth_adapter_eval.py \
  --model models/Qwen3.5-2B \
  --adapter checkpoints/q1_2b/adapter_final \
  --data data/splits_stage_q1_withreal/test.jsonl \
  --gen-batch-size 8 --out-dir checkpoints/q1_2b/eval_test --prefix test

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python export/merge_unsloth_lora.py \
  --base models/Qwen3.5-2B \
  --adapter checkpoints/q1_2b/adapter_final \
  --out checkpoints/q1_2b_merged
```

---

## 5. GRPO_Q1：探针 → 组数据

探针用**与训练相同的 q1v2 reward** 给 G=8 rollout 打分；`regular` 场景由 `val_slice` 支持。

```bash
uv run python eval/run_grpo_value_probe.py \
  --src data/splits_stage_q1_withreal/train.jsonl \
  --model models/Qwen3.5-2B \
  --adapter checkpoints/q1_2b/adapter_final \
  --out checkpoints/grpo_value_probe_v2 \
  --reward-set q1v2 \
  --scenes regular,multi_frag,semantic_group --include-empty \
  --g 8 --temperature 1.2 --top-p 1.0 --max-new-tokens 256 \
  --max-seq-length 8192 --max-pixels 8294400 --num-workers 8

uv run python data/scripts/compose_grpo_q1_v2.py
```

`compose_grpo_q1_v2.py` 的筛选（配比见 `data/recipes/grpo_q1_v2.yaml`）：

- `good` = reward ≥ `good_frac`(0.8) × 场景 ceiling（正例 1.0 / 负例 2.5）；
- 硬门：`0.125 ≤ frac_good ≤ 0.875` 且 `std ≥ 0.03`；
- 排序：正例按 `over+under+0.5*empty_miss`，负例按 `1-2|frac_good-0.5|`；
- 产出 500 条：`empty 40% / multi_frag 25% / semantic_group 25% / regular 10%`。

> ❗ 旧的 `max R > 0.5` 已废弃：新 reward 正例 ceiling 1.0、负例 2.5，固定阈值语义不一致。

---

## 6. GRPO_Q1 v2（单轮）

```bash
uv run python train/unsloth_grpo.py \
  --data data/splits_grpo_q1_v2/train.jsonl \
  --val  data/splits_grpo_q1_v2/val.jsonl \
  --model models/Qwen3.5-2B \
  --adapter checkpoints/q1_2b/adapter_final \
  --run-id grpo_q1_v2 \
  --reward-set q1v2 \
  --num-generations 8 --temperature 1.0 --top-p 1.0 --beta 0.04 \
  --max-completion-length 256 --gen-chunk-size 2 \
  --batch-size 8 --grad-accum 4 \
  --lr 5e-6 --epochs 1.0 --lora-rank 16 --lora-alpha 32 \
  --save-steps 50 --logging-steps 5 \
  --max-seq-length 8192 --max-pixels 8294400 --num-workers 4 \
  --max-grad-norm 1.0 --optim adamw_8bit --seed 42
```

> `--adapter` + 不加 `--continue-adapter` = 把 SFT adapter 烘成**冻结 π_ref**，新挂 r16 LoRA（与历史 GRPO_1 一致）。
> 2B × G=8 × 大图 generate 是显存峰值：先 `--gen-chunk-size 1`，OOM 再降到 `--num-generations 4`。

merge 出 Q1 里程碑底模：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python export/merge_unsloth_lora.py \
  --base checkpoints/q1_2b_merged \
  --adapter checkpoints/grpo_q1_v2/adapter_final \
  --out checkpoints/q1_2b_grpo_merged
```

---

## 7. SFT_Q2（在 Q1 里程碑底模上）

> 底模用 **`checkpoints/q1_2b_merged`**（SFT_Q1）——实测 GRPO_Q1 会掉点，已丢弃（见 `docs/GRPO_Q1_V2.md` §6）。
> 实测配置：`--num-workers 0 --no-pin-memory`（4 worker 时每个 worker COW 复制 ~2.6GB，会把 15GB 内存吃满）、
> `--post-eval-batch-size 2`、`--max-seq-length 8192`。

```bash
uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q2_ocr_mt_v2/train.jsonl \
  --val  data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --model checkpoints/q1_2b_merged \
  --out-root checkpoints --run-id q2_2b \
  --max-seq-length 8192 \
  --batch-size 2 --grad-accum 8 \
  --lr 1.5e-5 --epochs 1.0 \
  --lora-rank 32 --lora-alpha 64 \
  --save-steps 500 --eval-steps 250 --logging-steps 10 --eval-batch-size 2 \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4 --tf32 \
  --warmup-ratio 0.03 --weight-decay 0.01 --lr-scheduler cosine \
  --optim adamw_8bit --max-grad-norm 1.0 \
  --post-eval-batch-size 8 --seed 42
```

**不带** `--finetune-vision` → 冻结 vision（保留 Q1 学到的点位感知）。post-eval 会自动跑满 val 并批量生成。

---

## 8. Q2 评测 + merge

```bash
uv run python eval/run_q2_eval.py \
  --predictions checkpoints/q2_2b/metrics/final_predictions.jsonl \
  --split data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --out-dir checkpoints/q2_2b/metrics --prefix q2

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python export/merge_unsloth_lora.py \
  --base checkpoints/q1_2b_merged \
  --adapter checkpoints/q2_2b/adapter_final \
  --out checkpoints/q2_2b_merged
```

门禁（[`Q2_PLAN.md`](Q2_PLAN.md) §2）：`source_hit ≥ 0.88`、`effective_empty ≥ 0.85`、`hallucination ≤ 0.10`、`xml_well_formed ≥ 0.97`、`format_leak = 0`。不达标再进 §9。

---

## 9. GRPO_Q2（可选，单轮）

> 经验：2B SFT 之后 GRPO 信号极少（Q1 实测 25%），且 8GB+WSL 跑不了全分辨率 GRPO。
> 除非有 ≥8MP 可用的 24GB 卡，否则**不建议**做这一步。

```bash
uv run python eval/run_grpo_value_probe.py \
  --src data/splits_stage_q2_ocr_mt_v2/train.jsonl \
  --model checkpoints/q2_2b_merged \
  --adapter checkpoints/q2_2b/adapter_final \
  --out checkpoints/grpo_value_probe_q2 \
  --reward-set q2 \
  --scenes regular,multi_frag,semantic_group --include-empty \
  --g 8 --temperature 1.2 --max-new-tokens 512 \
  --max-seq-length 8192 --max-pixels 8294400 --num-workers 8

uv run python data/scripts/compose_grpo_q2_ocr_mt.py

uv run python train/unsloth_grpo.py \
  --data data/splits_grpo_q2_ocr_mt/train.jsonl \
  --val  data/splits_grpo_q2_ocr_mt/val.jsonl \
  --model checkpoints/q2_2b_merged \
  --adapter checkpoints/q2_2b/adapter_final \
  --run-id grpo_q2_ocr_mt \
  --reward-set q2 \
  --num-generations 8 --temperature 1.0 --top-p 1.0 --beta 0.04 \
  --max-completion-length 512 --gen-chunk-size 1 \
  --batch-size 8 --grad-accum 4 \
  --lr 5e-6 --epochs 1.0 --lora-rank 16 --lora-alpha 32 \
  --save-steps 50 --logging-steps 5 \
  --max-seq-length 8192 --max-pixels 8294400 --num-workers 4 \
  --max-grad-norm 1.0 --optim adamw_8bit --seed 42

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python export/merge_unsloth_lora.py \
  --base checkpoints/q2_2b_merged \
  --adapter checkpoints/grpo_q2_ocr_mt/adapter_final \
  --out checkpoints/q2_2b_grpo_merged
```

---

## 10. Studio 探伤 + 导出

```bash
BASE=checkpoints/q2_2b_merged \
bash eval/marker_studio/run_q2.sh
#   探伤 http://127.0.0.1:7865      （面积预设已换成 0.08% / 0.14% / 0.2%）
#   标注 http://127.0.0.1:7865/label

# 达标后导出（若跑了 §9，最终整模就是 checkpoints/q2_2b_grpo_merged）
bash export/export_gguf.sh checkpoints/q2_2b_merged \
  exports/Qwen35-2B-Point-OCR-MT-Q4_K_M.gguf Q4_K_M
```

---

## 显存 / 时间预算（单卡 8GB）

| 阶段 | 关键设置 | 峰值显存 | 预计 |
|---|---|---|---|
| SFT_Q1 | 2B 4bit, 2×8, 8192, vision LoRA | ~4.5–5GB | ~13–16h |
| GRPO_Q1 | G=8, gen-chunk 2→1 | 峰值（generate） | 数小时–1 天 |
| SFT_Q2 | 2B 4bit, 2×8, 8192, 冻 vision | ~3.9GB | ~17h + eval ~3h |
| GRPO_Q2 | G=8, completion 512, gen-chunk 1 | 峰值 | ~1 天 |

OOM 时的降级顺序：`--gen-chunk-size 1` → `--num-generations 4` → `--batch-size 2 --grad-accum 16`（SFT）→ 去掉 `--finetune-vision`。

## 冒烟（正式开跑前建议各跑一次）

```bash
# SFT 冒烟：不写进正式 run-id
uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q1_withreal/train.jsonl --val data/splits_stage_q1_withreal/val.jsonl \
  --model models/Qwen3.5-2B --out-root /tmp/smoke --run-id smoke \
  --max-seq-length 8192 --batch-size 2 --grad-accum 8 --lora-rank 32 --lora-alpha 64 \
  --max-samples 16 --max-steps 4 --skip-post-eval --num-workers 4

# GRPO 冒烟
uv run python train/unsloth_grpo.py \
  --data data/splits_grpo_q1_v2/train.jsonl --val data/splits_grpo_q1_v2/val.jsonl \
  --model models/Qwen3.5-2B --adapter checkpoints/q1_2b/adapter_final \
  --out /tmp/smoke_grpo --reward-set q1v2 --max-samples 8 --max-steps 2 --skip-post-eval
```

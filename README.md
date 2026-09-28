# Point-conditioned OCR（Unsloth）

截图上画固定品红 X，模型只输出准星所在的最小语义块；空白 / 壳体 / 表格 / 纯图输出空字符串。
**Q2** 在同一几何上再要求 **英文源文 + 简体中文翻译**（XML）。

- 训练栈：**Unsloth `FastVisionModel` + TRL**（不走坐标条件、不用 LLaMA-Factory、只用 [uv](https://docs.astral.sh/uv/)）
- **部署目标：消费级终端 → `models/Qwen3.5-0.8B`**（0.8B 极限路线，见 [`docs/PIPELINE_QWEN35_08B.md`](docs/PIPELINE_QWEN35_08B.md)）
- 现有权重：2B 两轮 SFT 已完成（`checkpoints/q2_2b_merged`）；0.8B 干净基线待跑；4B teacher 本地可跑 SFT
- 路线评估（转 0.8B / 4B / OPD）：[`docs/STRATEGY_REVIEW_4B_OPD.md`](docs/STRATEGY_REVIEW_4B_OPD.md)
- 平台：Linux + NVIDIA（本项目在 **WSL2 + RTX 4060 Laptop 8GB / 15GB RAM** 上跑通）

```bash
uv sync --extra dev
uv sync --extra train
uv pip install unsloth
uv run pytest -q          # 148 passed
```

AutoDL 弱网镜像：[`docs/AUTODL.md`](docs/AUTODL.md)。

---

## 一、工作区状态（2026-09-28）

> **换工作区先看 [`docs/HANDOVER_2026-09-28.md`](docs/HANDOVER_2026-09-28.md)**（现状/结论/下一步/文档地图）。
> 本节及以下保留工程细节，部分数字是 768px 旧口径。

**现在：2B 全流程已完成可部署；0.8B 在 2880² 分辨率上重跑中；GRPO/OPD 链路已就绪待跑。**

| 项 | 状态 |
|--|--|
| 2B Q1+Q2 | ✅ `checkpoints/q2_2b_merged`（block_hit 0.9396 / source 0.9140 / chrF 0.3922；`both_hit 0.1725` 未过门禁） |
| 0.8B Q1+Q2 | ⚠️ 已有 768px 口径产物（`q2_08b_merged`）；**2880² 重跑进行中**（分辨率修复后） |
| 分辨率修复 | ✅ `--collator-resize max`（默认）+ 视觉 token 预算告警；train/eval/Studio 已对齐 |
| 纯文本样本 | ✅ 代码支持（两阶段翻译备用） |
| GRPO | ✅ 链路+数据（`data/splits_grpo_{q1,q2}_hq200/`，200/60）+ 信号门禁；**未开跑** |
| OPD | ✅ TRL `GKDTrainer` + VL 适配层 `train/unsloth_gkd_vl.py`，本地 `--selfcheck` 已通过；**未真跑** |
| 4B | ✅ 基座已下载；本地 SFT 可跑（~1.74 s/sample，43h）；GRPO/OPD 需 32GB |
| 测试 | ✅ 180 passed |

| | |
|--|--|
| **当前可部署权重** | **`checkpoints/q2_2b_merged`**（Qwen3.5-2B + Q1 SFT + Q2 SFT，整模、无 LoRA） |
| Q1 里程碑权重 | `checkpoints/q1_2b_merged`（Qwen3.5-2B + Q1 SFT） |
| 提示词 | Q1：`a2_v3`（只出 Markdown 块）；**Q2：`ocr_mt_v1`**（`<source>` / `<translation>`） |
| 准星 | **`x45r`：每张图独立均匀随机，面积占比 0.08%–0.2%**，α=255（尺寸鲁棒性由 SFT 数据承担，**不是** GRPO 任务） |
| 数据划分 | Q1 `data/splits_stage_q1_withreal/`；Q2 `data/splits_stage_q2_ocr_mt_v2/` |
| Occupancy 门禁 | 页墨占比 [25%, 70%]；正例点取块 inner 80% 的**五区 diamond** 采样 |

**硬约束：** 不破坏 Q1 跟点 OCR；空例 GT 必须是 `""`（不要空 XML）；**不要自动开训**，等入口命令被显式执行。

### 1.1 已经做完的（按时间顺序）

1. **准星协议改为 `x45r`**：每图随机 `Uniform(0.08%, 0.2%)` 面积，写入 `metadata.marker_area_frac`。
   覆盖 `build_point.py` / `pools/emit.py` / `compose_q1_withreal.py` / `build_real_label_pool.py` /
   `build_real_mt_pool.py` / `remake_pool_markers.py` / `build_pool.py` / studio。
   **全部池已用新协议重渲**。
2. **数据重建**：Q1 五池 + Q2 五池 + 真实池（`real_labeled` 1526，`real_labeled_mt` 含翻译）。
   补了两个 chrome topup（`topup_q1_chrome.sh` 新建、`topup_q2_chrome.sh` 复用），因为池的 chrome 比例达不到 compose 的 40% 配额。
3. **SFT_Q1（2B）** — `checkpoints/q1_2b/`，2397 步，best `eval_loss 0.0171`。**大幅超过门禁**：
   `block_hit 93.96%`（旧 0.8B 87.6%）、`empty 98.8%`、`multi_frag 85.6%`（旧 75.2%）、`semantic_group 97.1%`（旧 76.6%）。
4. **GRPO_Q1 v2** — 完整尝试后**判定失败并丢弃**：探针 503 条只有 127 条有学习信号（25%），
   最终 110 条训练；且 8GB+WSL 跑不了全分辨率 GRPO（被迫降到 1440²/4096），结果全面变差
   （hit −2.4pp、semantic_group −5.8pp）。**结论：Q1 里程碑用纯 SFT_Q1**。详情 [`docs/GRPO_Q1_V2.md`](docs/GRPO_Q1_V2.md) §6。
5. **SFT_Q2（2B）** — `checkpoints/q2_2b/`，3101 步，best `eval_loss 0.1582`。
   **Q2 v1 的灾难级问题被解决**：`effective_empty 65.8% → 99.6%`、`hallucination 34.2% → 0.37%`、
   `source_hit 89.3% → 91.4%`、`translation_chrF 0.341 → 0.392`。
6. **工程加固**（都是踩坑后加的）：
   - `--resume-from-checkpoint`（SFT + GRPO）——WSL 重启后可从 Trainer checkpoint 精确续训；
   - **修复 `--finetune-vision`**：在 Qwen3.5 上原本是 no-op（vision LoRA 一直可训练），现在真正生效；
   - `POINT_OCR_GC=standard`——用重算式 gradient checkpointing，避免 Unsloth 把激活 offload 到宿主内存（GRPO 曾因此 OOM）；
   - post-eval 批量生成 + **持久性** OOM 降 batch；`eval/run_merged_eval.py` 评整模；探针支持 `--scene-mix`、`--reward-set`。

### 1.2 结果（全量/分层 val）

**Q2（`checkpoints/q2_2b`，val 2684 全量，`eval/run_q2_eval.py`）**

| 指标 | 门禁 | Q2 v2（2B） | Q2 v1（0.8B） |
|---|---|---|---|
| `source_hit_rate` | ≥0.88 | **0.9140** ✅ | 0.893 |
| `translation_chrf` | ≥0.34 | **0.3922** ✅ | 0.341 |
| `both_hit_rate` | ≥0.20 | 0.1725 ❌（差 2.7pp） | 0.108 |
| `xml_well_formed_rate` | ≥0.97 | **0.9883** ✅ | 0.984 |
| `unclosed_translation_rate` | ≤0.01 | **0.0014** ✅ | 0.015 |
| `effective_empty_rate` | ≥0.85 | **0.9963** ✅ | 0.658 |
| `hallucination_rate` | ≤0.10 | **0.0037** ✅ | 0.342 |
| `plain_text_rate` | — | **0.0000** ✅ | 0.091 |
| `format_leak_rate` | 0 | 0.0004（1 条）≈ | 0 |

**Q1（`checkpoints/q1_2b`，500 条分层 val）**：`block_hit 0.9396`、`over 0.0020`、`empty 0.9882`、`mean_edit 0.9652`。
按场景：regular 0.971 / semantic_group 0.971 / multi_frag 0.856 / empty 0.988。

### 1.3 已知问题 / 局限

1. **翻译仍是弱项**（`both_hit` 未达 0.20）。chrF 是**单参考**指标，合理意译会被判低分，真实质量应好于该数字；但 2B 确实不是高质量 MT 量级。
2. **`multi_frag`（跨 fragment 整段）是 Q1 唯一剩余弱项**（85.6%），GRPO 没能改善。
3. **GRPO 在本地不可行**：2B × 8.3MP × G≥4 会触发 WSL 驱动 `make_resident ENOMEM` / `CUDA device not ready`；且 2B SFT 之后 signal 样本太少（Q1 实测 25%）。
4. **参考译文有 ~1.9% 含繁体字**（真实/OPUS 语料残留，0.6% 比例 >5%）——是数据噪声，不是模型输出。
5. **偶发截断**：约 0.14% 的正例只出 `<source>` 就停（全量 val 3/2139）。

### 1.4 后续计划

**本文件不再维护计划，已迁到 [`docs/HANDOVER_2026-09-28.md`](docs/HANDOVER_2026-09-28.md) §6 与 §7。**
简版：P0 0.8B@2880² 跑完看分辨率红利 → P1 0.8B GRPO（HQ-200） → P2 4B SFT → P3 4B GRPO（定位优先）
→ P4 OPD（teacher=4B-GRPO-Q2）→ P5 翻译单独用数据/两阶段解决。

---

## 二、入口地图

### 训练（显式开跑）

| 入口 | 做什么 | 数据 | 底 / adapter | 产出 |
|---|---|---|---|---|
| `train/unsloth_stage_a.py` | **Unsloth SFT 通用脚本**（Q1/Q2 都用它） | `--data` / `--val` | `--model`（可 `--resume-from-checkpoint`） | `checkpoints/<run-id>/` |
| `train/run_08b_pipeline.sh` | **0.8B 两轮 SFT 一键流水线**（对齐 2B 口径；`smoke`/`q1`/`merge-q1`/`q2`/`merge-q2`/`eval-*`） | 同 2B 数据 | `models/Qwen3.5-0.8B` | `checkpoints/q{1,2}_08b{,_merged}/` |
| `train/unsloth_grpo.py` | GRPO 通用脚本，`--reward-set {q1,q1v2,q2}` | `--data` / `--val` | `--adapter` + `--continue-adapter` | 同上 |
| `train/run_unsloth_grpo.sh` / `train/run_unsloth.sh` / `train/run_unsloth_q2.sh` | 历史包装脚本（Q1/Q2 v1 配方，保留作参考） | — | — | — |

**当前 2B 配方**（详细命令见 [`docs/PIPELINE_QWEN35_2B.md`](docs/PIPELINE_QWEN35_2B.md)）：

- SFT_Q1：`--model models/Qwen3.5-2B --max-seq-length 8192 --lr 5e-5 --epochs 1.5 --lora-rank 32 --lora-alpha 64 --finetune-vision`
- SFT_Q2：`--model checkpoints/q1_2b_merged --max-seq-length 8192 --lr 1.5e-5 --epochs 1.0 --lora-rank 32 --lora-alpha 64`（**不开** `--finetune-vision`，冻结 vision）

完整超参以 `checkpoints/<run>/run_config.json` 为准。

### 数据

| 入口 | 做什么 | 产出 |
|---|---|---|
| `data/scripts/build_q1_pools_sequential.sh` / `build_q2_pools_sequential.sh` | 顺序渲池（新标记） | `data/pools_q/`、`data/pools_q2/` |
| `data/scripts/topup_q1_chrome.sh` / `topup_q2_chrome.sh` | 强制补 chrome 页（配额不够时） | 追加进对应池 |
| `data/scripts/build_real_label_pool.py` / `build_real_mt_pool.py` | 真实截图重打标 + 克制补中文 | `data/pools_real/real_labeled{,_mt}/` |
| `data/scripts/compose_q1_withreal.py` | Q1 划分 | `data/splits_stage_q1_withreal/` |
| `data/scripts/compose_q2_ocr_mt.py` | Q2 划分（2× Q1 配比） | `data/splits_stage_q2_ocr_mt/` |
| `data/scripts/filter_q2_negatives.py` | 清洗误导负例（prose 表格 / clear-on-ink / chrome 错标）+ 配平 | `data/splits_stage_q2_ocr_mt_v2/` |
| `data/scripts/compose_grpo_q1_v2.py` / `compose_grpo_q2_ocr_mt.py` | 旧版按场景配比组 GRPO 数据（产物已归档，新流程见下一行） | `data/splits_grpo_q1_v2/` |
| `data/scripts/build_grpo_candidates.py` | **HQ-200 候选池**（弱点桶分层 + 页/模板/尺寸多样性 + dHash 去重） | `data/splits_grpo_cand/` |
| `data/scripts/compose_grpo_q1_hq200.py` | **信号门禁**组 200/60（`--kind q1|q2`；`reward_max_frac≥0.95` 且 `std≥0.10`） | `data/splits_grpo_{q1,q2}_hq200/` |
| `bash train/run_grpo_hq.sh` | **GRPO 通用一条龙**（`KIND=q1|q2 SIZE=08b|2b|4b`；status→cand→seed→probe→compose→dryrun→train→merge→eval） | `checkpoints/grpo_<kind>_<size>/` |
| `uv run python train/unsloth_gkd_vl.py --selfcheck …` | **OPD**（TRL GKDTrainer 视觉适配层；`--selfcheck` 验逻辑） | `checkpoints/gkd_opd/`（真训练时） |
| `data/scripts/build_pool.py` | 单池渲图（`--marker-area-min/max`） | `data/pools_*/<pool>/` |

### 探伤 / 评测 / 导出

| 入口 | 做什么 |
|---|---|
| `bash eval/marker_studio/run_q2.sh` | **探伤/标注台**（默认 `--base checkpoints/q2_2b_merged`，无 LoRA） |
| `bash scripts/pack_for_cloud.sh --dry-run` | **上云打包**：生成代码/图片清单（`dist/cloud_pack/`），可出 tarball |
| `uv run python scripts/relocate_paths.py --old-root <旧根> --apply` | **换机后重定位** split 里的绝对图片路径 |
| `eval/run_q2_eval.py` | **Q2 专用指标**（source/chrF/XML/empty），读已有 predictions，无需 GPU |
| `eval/run_unsloth_adapter_eval.py` | 评 **LoRA**（`--model <底> --adapter <dir>`） |
| `eval/run_merged_eval.py` | 评**整模**（`--model <merged dir>`，无 LoRA 参数） |
| `eval/run_grpo_value_probe.py` | G=8 rollout 探针（`--reward-set`、`--scene-mix`、`--max-samples`） |
| `export/merge_unsloth_lora.py` | LoRA bake 进本地底模 |
| `export/export_gguf.sh` | merge 后量化 GGUF |

---

## 三、磁盘上有什么

### 权重

| 路径 | 含义 |
|---|---|
| `models/Qwen3.5-2B` | 2B 基座（Instruct，2.26B） |
| `models/Qwen3.5-0.8B` | **0.8B 基座**（消费级终端部署目标） |
| `models/Qwen3.5-4B` | 4B 基座（teacher 分支，需 ≥24GB 卡做 GRPO/OPD） |
| `checkpoints/q1_2b/adapter_final` | SFT_Q1 LoRA（best eval_loss 0.0171） |
| **`checkpoints/q1_2b_merged/`** | **Q1 里程碑整模**（SFT_Q1；SFT_Q2 的底模） |
| `checkpoints/q2_2b/adapter_final` | SFT_Q2 LoRA（best eval_loss 0.1582） |
| **`checkpoints/q2_2b_merged/`** | **可部署 2B 整模（Q1 SFT → Q2 SFT）** |
| `checkpoints/q1_08b/`、**`checkpoints/q1_08b_merged/`** | **0.8B SFT_Q1 + 整模**（block_hit 0.9034） |
| `checkpoints/q2_08b/`、**`checkpoints/q2_08b_merged/`** | **0.8B SFT_Q2 + 可部署整模**（Q2 source_hit 0.8303，**未过门禁**；见 [`eval/results/q08b_stage_report.md`](eval/results/q08b_stage_report.md)） |
| `checkpoints/q2_08b/eval_val/metrics/` | 0.8B Q2 全量 val 报告 + 2684 条预测 |
| `checkpoints/q1_withreal/`、`q2_ocr_mt/` | 旧 **0.8B** 路线**仅保留 metrics/ 与 run_config**（权重已于 2026-09-26 清理） |

> **2026-09-26 清理**：删除了已证伪的 GRPO 产物（`grpo_q1_v2`、`q1_2b_grpo_merged`）、
> 旧 0.8B GRPO 系列（`grpo_q1_signal`/`grpo_q2_marker`/`q1_grpo_merged`/`grpo_q1_hard`）、
> 旧 0.8B 合模（`q1_withreal_merged`）与旧 0.8B Q1 权重目录，checkpoints 25G → 12G。
> 结论见 `docs/GRPO_Q1_V2.md` §6；0.8B 旧数字不可用于判断能力（见 `docs/STRATEGY_REVIEW_4B_OPD.md` §1.3）。

### 数据划分

| 路径 | 阶段 | 提示 | 规模 |
|---|---|---|---|
| `data/splits_stage_q1_withreal/` | Q1 SFT（2B，已训） | `a2_v3` | train 25562 / val 1421 / test 1418 |
| `data/splits_stage_q2_ocr_mt_v2/` | Q2 SFT（2B，已训） | `ocr_mt_v1` | train 49616 / val 2684 / test 2666 |
| `data/splits_grpo_q1_v2/` | GRPO_Q1 v2（110 条，实验） | `a2_v3` | train 110 / val 80 |
| `data/splits_grpo_{q1_signal,q2_marker,q1_hard}/` | 旧 0.8B GRPO 划分 | `a2_v3` | 归档 |
| `data/_archive_v1/` | 重渲前的旧划分与池 jsonl 备份 | — | — |

Q2 正例 assistant 形如：

```xml
<source>
…Markdown…
</source>
<translation>
…简体中文…
</translation>
```

空例 assistant 是空字符串。每行 `metadata.marker_area_frac ∈ [0.0008, 0.002]`。

### 评测报告

| 路径 | 内容 |
|---|---|
| `checkpoints/q1_2b/eval_val/metrics/` | SFT_Q1 500 条分层 val 报告 |
| `checkpoints/q2_2b/metrics/{final_report,q2_report}.{json,md}` | SFT_Q2 全量 val 整串 + 拆分报告 |
| `checkpoints/q2_2b/metrics/final_predictions.jsonl` | 2684 条逐条预测（可离线重打分） |
| `eval/results/q1_2b_stage_report.md` / `q2_2b_stage_report.md` | **阶段总结报告** |
| `eval/results/marker_size_preview_*.png` | 准星尺寸对比预览 |
| `eval/results/qwen2b_feasibility/` | 2B 迁移可行性实测（显存/速度） |
| `eval/results/q2_sft_analysis/` | Q2 v1 诊断证据（对照实验、drift、截图） |

---

## 四、探伤 / 标注台

同一进程两个页面。整模走 `--base`；LoRA 走 `--adapter`（会读 `adapter_config` 里的底）。
传 `--base` 且不传 `--adapter` 时默认=无 LoRA。

```bash
bash eval/marker_studio/run_q2.sh                 # 默认：q2_2b_merged，无 LoRA

uv run python eval/marker_studio/server.py \
  --base checkpoints/q2_2b_merged                 # 等价手写
```

| 台 | URL | 用途 |
|---|---|---|
| 探伤 | http://127.0.0.1:7865 | 自己放十字、改面积/解码/提示词，立刻跑当前权重 |
| 标注 | http://127.0.0.1:7865/label | 真实截图画框、写块文案 |

- 提示词预设：**`ocr_mt_v1`**（Q2）；`a2_v3` 可单独验证 Q1 跟点。
- 面积预设：新协议 `0.08% / 0.14% / 0.2%` + 旧 `0.425% / 0.5%`。
- 预处理与训练一致：`resize_for_ovis` = `448²–2880²`；解码走与评测同一个 `generate_point_text`。

---

## 五、评测

**Q2 拆分指标**（无需 GPU）：

```bash
uv run python eval/run_q2_eval.py \
  --predictions checkpoints/q2_2b/metrics/final_predictions.jsonl \
  --split data/splits_stage_q2_ocr_mt_v2/val.jsonl
# → checkpoints/q2_2b/metrics/q2_report.{json,md}
```

关键指标：`source_hit_rate`、`translation_chrf`、`effective_empty_rate`、`hallucination_rate`、
`xml_well_formed_rate`、`both_hit_rate`。定义与门禁见 [`docs/Q2_PLAN.md`](docs/Q2_PLAN.md)。

**评测加速**：post-eval 已批量生成（按面积排序 + 持久性 OOM 降 batch）。RTX 4060 8GB 上
batch-1 ~0.10 samp/s → batch-8 ~0.45（**~4×**）；结构性指标不漂移（证据
`eval/results/q2_sft_analysis/eval_batch_drift.txt`）。**对比模型时务必用同一 batch size。**

---

## 六、工程注意事项（踩过的坑）

| 事项 | 结论 |
|---|---|
| **宿主内存** | 15GB 很紧。DataLoader `--num-workers>0` 会因 fork COW 每 worker 复制 ~2.6GB（4 worker ≈ 10.4GB）→ **长跑用 `--num-workers 0 --no-pin-memory`**（RAM 稳定 ~7GB）。GRPO 还会触发 Unsloth 把激活 offload 到宿主内存 → 用 `POINT_OCR_GC=standard`。 |
| **GPU 8GB + WSL** | 8.3MP × G≥4 的 GRPO 会 `make_resident ENOMEM` / `CUDA device not ready`。只能降分辨率或换卡。 |
| **序列长度** | 受两件事约束：① 训练时 collator 曾把图像宽度静默压到 768（→ 实际序列仅 ~350–810 token，4096 都够）；② **2026-09-28 修复后**由 `--max-pixels` 决定，视觉 token ≈ `max_pixels/1024`。规则：`max_pixels/1024 + ~2048 ≤ --max-seq-length`。默认 `2048²`+`8192`；`2880²` 需 `≥12288`。详见下方 2026-09-28 修复。 |
| **`--finetune-vision`** | 曾经是 no-op；现在真正生效。SFT_Q1 开（标记几何变了），SFT_Q2 不开（冻结，保住点位感知）。 |
| **断线恢复** | `--resume-from-checkpoint checkpoints/<run>/checkpoint-N`（配合同一 `--run-id`），恢复优化器/调度器/RNG。 |
| **长任务日志** | 用 `setsid nohup ... > logs/<name>.log 2>&1 < /dev/null &`，别放 `/tmp`（重启会清）。 |
| **训练/评测分辨率必须一致** | 两边都由 `--max-pixels` 决定（视觉 token ≈ `max_pixels/1024`）。train、`eval/run_*.py`、`marker_studio` 默认都是 `2048²`；改训练就需要同步改评测，否则 OOD。 |

### 2026-09-28 修复（两个地基问题）

1. **图像被静默压到 768px 宽**（像素量损失 ~23×，`--max-pixels` 形同虚设）
   - 根因：`UnslothVisionDataCollator(model, tokenizer)` 默认 `resize="min"` + `_ensure_vision_image_size()` 把
     `vision_config.image_size` 设为 **768**，collator 按宽度裁剪。实测 3760×2112 → `image_grid_thw [1,28,48]`（=768×448）。
   - 后果：训练在 ~336 视觉 token，而**评测/Studio 走 processor 无此上限（全分辨率）→ train/eval 分辨率不一致**。
   - 修复：`unsloth_stage_a.py` 新增 `--collator-resize {max,min}`，**默认 `max`（不裁剪）**；分辨率由 `--max-pixels` 控制；
     新增视觉 token 预算告警。评测脚本/Studio 默认同步为 `2048²`。
   - 最密页 batch=2 forward+backward 实测（0.8B）：

     | max_pixels | max_seq | image tokens | peak reserved |
     |---|---|---|---|
     | 1280² | 8192 | 1564 | 2192 MiB |
     | **2048²（默认）** | 8192 | 4015 | **3406 MiB** |
     | 2880² | 12288 | 7854 | 5790 MiB |

2. **样本必须带图**（无法训练/评测纯文本任务，如“只翻译”）
   - Unsloth collator **原生支持**混批（实测 image+text 混批 forward+backward，468/468 LoRA 张量有梯度）；
     限制只在我们的转换层。
   - 修复：`sharegpt_to_unsloth` / `run_final_generate_eval` 保留无图行；`point_ocr.infer` 支持 `image=None`
     （`generate_point_text` / `generate_point_batch` 均支持混批）。见 `tests/test_text_only_support.py`。
   - 纯文本评测：predictions JSONL 里 `images: []`（或无该键）即可，`run_q2_eval.py` 照常打分。

> 本目录所有 2026-09-28 之前的实验（0.8B/2B）都是在 **768px 宽 + 全分辨率评测**下得到的，
> 数字偏低且 train/eval 不一致；分辨率修复后应重跑基线再对比。

---

## 七、其它文档

| 文档 | 何时看 |
|---|---|
| [`docs/HANDOVER_2026-09-28.md`](docs/HANDOVER_2026-09-28.md) | **换工作区先看这份**（现状/结论/下一步/资产/坑/文档地图） |
| [`docs/CLOUD_RUN_32GB.md`](docs/CLOUD_RUN_32GB.md) | 32GB 云端训练命令（0.8B/4B、GRPO） |
| [`docs/OPD_PLAN.md`](docs/OPD_PLAN.md) | **OPD 方案 + 已实现的 VL 适配层 + selfcheck 结果** |
| [`docs/PIPELINE_QWEN35_2B.md`](docs/PIPELINE_QWEN35_2B.md) | **2B 全流程完整开训命令（重渲 → 导出）** |
| [`docs/PIPELINE_QWEN35_08B.md`](docs/PIPELINE_QWEN35_08B.md) | **0.8B 干净基线（对齐 2B 口径）命令 + 旧 0.8B 污染项清单** |
| [`docs/STRATEGY_REVIEW_4B_OPD.md`](docs/STRATEGY_REVIEW_4B_OPD.md) | **转 0.8B / 4B / OPD 的路线评估（含 4B 本地实测）** |
| [`docs/CLOUD_PACKAGING.md`](docs/CLOUD_PACKAGING.md) | **打包上云（清单 / rsync / 路径重定位 / 云端自检）** |
| [`docs/Q2_PLAN.md`](docs/Q2_PLAN.md) | Q2 复盘 / 指标定义 / 门禁 / 训练计划 / `--finetune-vision` 与 GRPO 复盘 |
| [`docs/GRPO_Q1_V2.md`](docs/GRPO_Q1_V2.md) | GRPO_Q1 v2 数据标准 / reward / 运行顺序 / **失败结论** |
| [`docs/GRPO_Q1_HQ200.md`](docs/GRPO_Q1_HQ200.md) | **再试 GRPO**：信号优先的 HQ-200 数据 + 训练链路 + 门禁 |
| [`docs/MARKER_SPEC.md`](docs/MARKER_SPEC.md) | 准星几何（`x45r` 0.08%–0.2% + 旧协议） |
| [`docs/PROMPTS.md`](docs/PROMPTS.md) | `a2_v3` / `ocr_mt_v1` |
| [`docs/POOL_PIPELINE.md`](docs/POOL_PIPELINE.md) | 合成池 / compose |
| [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) | 数据流与训练分层 |
| [`docs/AUTODL.md`](docs/AUTODL.md) | 云主机镜像 / uv / Playwright |
| [`docs/UNSLOTH.md`](docs/UNSLOTH.md) | Unsloth 安装冒烟 |
| [`export/README.md`](export/README.md) | merge → GGUF |
| `docs/CURRICULUM_PLAN*.md`、`docs/ROUND_B_PLAN.md` | **已废弃**（旧的 A1/A2 / OvisOCR2 路线，仅存档） |

---

## 八、归档：0.8B 路线（旧 run，勿当基线）

早期在 `models/Qwen3.5-0.8B` 上完成：Q1 SFT（`q1_withreal`，hit 87.6%）→ GRPO_1（`grpo_q1_signal`）
→ GRPO_2（`grpo_q2_marker`）→ merge（`q1_grpo_merged`）→ Q2 SFT v1（`q2_ocr_mt`，empty 65.8%）。
Q2 v1 的失败诊断（prose 表格负例、`neg_clear` 落在 chrome/墨迹上、`--finetune-vision` no-op）
直接催生了 2B 这一轮的数据与工程改动。

**这些旧数字不可用作 0.8B 能力判断**（至少有 5 个混淆：seq 4096、旧大准星、未清洗负例、
底模是已证伪的 GRPO、`--finetune-vision` no-op），完整清单见
[`docs/PIPELINE_QWEN35_08B.md`](docs/PIPELINE_QWEN35_08B.md) §0。
权重已于 2026-09-26 清理，仅保留 `q1_withreal/` 与 `q2_ocr_mt/` 的 `metrics/` 与 `run_config.json` 作证据。
新 0.8B 用 `train/run_08b_pipeline.sh`。

Apache-2.0（本仓库代码）。基座与截图各自有许可；真实截图不要公开。

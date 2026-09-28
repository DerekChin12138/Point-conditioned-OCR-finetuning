# Q1 阶段报告（Qwen3.5-2B，执行到 SFT_Q2 前）

日期：2026-09-24 20:30。执行范围：**重渲数据 → SFT_Q1 → GRPO_Q1**。
**SFT_Q2 未开始**（按约定停在这里）。

---

## 0. 结论速览

| 项目 | 状态 | 是否达标 |
|---|---|---|
| 标记协议 `x45r` 0.08%–0.2% 随机 | 完成，全库重渲 | ✅ |
| Q1 SFT（2B, 2880², 8192 seq） | 完成 | ✅ **超预期** |
| GRPO_Q1 v2 | 完成但**全面变差** | ❌ **丢弃** |
| Q2 数据（新标记 + 清洗 v2） | 完成，待训 | ✅ 就绪 |
| SFT_Q2 | **未开始** | — |

**Q1 里程碑权重**：`checkpoints/q1_2b_merged`（= SFT_Q1，不含 GRPO）。

---

## 1. 标记协议变更

`x45r`：每张 marked 图独立 `Uniform(0.0008, 0.002)`（0.08%–0.2% 面积），α=255，标签写入
`metadata.marker_area_frac`。旧的 `x45c`(0.5%) / `x45d`(0.425%) 保留仅用于解析旧数据。

改动覆盖：`build_point.py` / `pools/emit.py` / `compose_q1_withreal.py` /
`build_real_label_pool.py` / `build_real_mt_pool.py` / `remake_pool_markers.py` /
`build_pool.py`（`--marker-area-min/max`）/ studio。双尺寸标记从 GRPO 任务中移除。

---

## 2. 数据重渲（全部用新标记）

| 池 | 产出 | 备注 |
|---|---|---|
| Q1 `semantic_group` | 2800 + 300 chrome topup | 新脚本 `data/scripts/topup_q1_chrome.sh`（原池只有 20% chrome，40% 配额不够） |
| Q1 `empty_special` | 3500 | |
| Q1 `empty_clear` | 5000 | |
| Q1 `multi_frag` | 2800 → compose 时 +2800 alt-fragment | |
| Q1 `core_inner` | 18000 | |
| Q2 五池 | 4200 / 5500 / 8000 / 8000 / 32000 | 中途因 WSL 重启中断，单独补跑 `core_inner` |
| Q2 chrome topup | +700 / +700 / +400 / +1600 | 40% chrome 配额 |
| 真实图 | 1526（`real_labeled`）+ MT 补译 144 条 API | |

划分：`splits_stage_q1_withreal` train 25562 / val 1421 / test 1418；
`splits_stage_q2_ocr_mt_v2` train 49616 / val 2684 / test 2666（负例清洗 + blank×1.35）。
所有行 `marker_area_frac ∈ [0.0008, 0.002]`（已校验）。

---

## 3. SFT_Q1（2B）——**达到并超过目标**

配置：`models/Qwen3.5-2B`，4bit QLoRA，`--finetune-vision`（已修复为真生效），
r32/α64，lr 5e-5，1.5 epoch，batch 2×accum 8，`max_seq_length 8192`，`max_pixels 8294400`。
2397 步，GPU 峰值 ~4.0GB，用时约 8h（含一次 WSL 重启后续训）。

**评测**（500 条分层 val，全分辨率 2880²，`eval/run_unsloth_adapter_eval.py`）：

| 指标 | 门禁 | 实测 | 旧 0.8B Q1 基线 |
|---|---|---|---|
| `block_hit_rate` | ≥0.88 | **0.9396** | 0.876 |
| `over_extraction_rate` | — | 0.0020 | 0.0056 |
| `empty_on_chrome_rate` | ≥0.95 | **0.9882** | 0.987 |
| `format_leak_rate` | 0 | 0 | 0 |
| `mean_edit_similarity` | — | 0.9652 | 0.9216 |

分场景：regular **0.971**、semantic_group **0.971**、multi_frag **0.856**、empty **0.988**。

对比旧 0.8B Q1 SFT：multi_frag 75.2%→**85.6%**、semantic_group 76.6%→**97.1%**、
real 82.9%→92.1%。best eval_loss 0.0171（step 2250），全程单调下降无过拟合。

---

## 4. GRPO_Q1 v2 ——**失败，已丢弃**

- 探针（G=8，q1v2 reward，`--scene-mix` 按配比）：503 条 → **127 signal（25%）**，
  其中 multi_frag 66 / semantic_group 49 / regular 8 / empty 4。
  → `empty` 几乎全饱和（reward 恒 2.5，std=0），**2B SFT 已把目标能力基本解决**。
- 组数据：110 条（mf 62 / sg 36 / reg 8 / empty 4）。
- 训练：28 步，lr 5e-6，`frac_reward_zero_std = 0.4`（40% 组无梯度），train reward 0.907→0.860。

**同 500 条 val 对比（全分辨率）**：

| 指标 | SFT_Q1 | +GRPO_Q1v2 | Δ |
|---|---|---|---|
| `block_hit_rate` | 0.9396 | 0.9154 | **−2.4pp** |
| `empty_on_chrome` | 0.9882 | 0.9763 | −1.2pp |
| `mean_edit_similarity` | 0.9652 | 0.9518 | −0.013 |
| semantic_group | 0.971 | 0.913 | **−5.8pp** |
| multi_frag | 0.856 | 0.844 | −1.2pp |

**原因**：
1. 信号不足（SFT 太强；empty 全饱和），有效样本只有 ~110 条；
2. **硬件被迫降分辨率**：8GB + WSL 跑不动 8.3MP×G≥4 的 GRPO
   （`dxgkio_make_resident: -12` → `CUDA driver error: device not ready`），
   实际在 **1440²/4096** 训练，而 SFT 是 2880²/8192 → 分布外优化。

**决定：不用 GRPO，Q1 里程碑 = `checkpoints/q1_2b_merged`。**

---

## 5. 事故与处置

| 事故 | 现象 | 处置 |
|---|---|---|
| WSL 重启 | SFT_Q1 在 step ~680 被杀，`/tmp` 清空 | 给两个训练脚本加 `--resume-from-checkpoint`，从 `checkpoint-500` 续训成功 |
| 评测极慢 | post-eval 在大图尾部 1 行/分钟（无 FlashAttention → O(n²)） | 改成**持久性**降 batch（原为每个 chunk 重试），并改用 bounded eval |
| 宿主 OOM | GRPO `shmem-rss 11–13GB` 撑爆 15GB RAM | GRPO `--num-workers` 默认改 0；新增 `POINT_OCR_GC=standard` 用标准重算式 checkpointing（不 offload 到内存） |
| CUDA device not ready | WSL GPU `make_resident` ENOMEM | GRPO 降到 1440²/4096 才跑通（这也导致第 4 节的结论） |

---

## 6. 当前产物

| 路径 | 说明 |
|---|---|
| `checkpoints/q1_2b/adapter_final` | SFT_Q1 LoRA（best eval_loss） |
| **`checkpoints/q1_2b_merged`** | **Q1 里程碑整模（SFT_Q1），SFT_Q2 的底模** |
| `checkpoints/grpo_q1_v2/` | GRPO 实验（**不推荐使用**） |
| `checkpoints/q1_2b_grpo_merged` | GRPO merge（**已证伪**） |
| `data/splits_stage_q1_withreal/` | Q1 SFT 数据（新标记） |
| `data/splits_stage_q2_ocr_mt_v2/` | **Q2 SFT 数据（就绪）** |
| `data/splits_grpo_q1_v2/` | GRPO 数据（110 条，实验用） |
| `checkpoints/q1_2b/eval_val/metrics/` | SFT_Q1 评测报告 |

---

## 7. 与预期的差距（如实汇报）

- **SFT_Q1 超预期**：hit 93.96% vs 门禁 88%、vs 旧 0.8B 87.6%；semantic_group 从 76.6% 提到 97.1%。
- **GRPO_Q1 未达预期（且为负收益）**：计划 500 条，实际只有 110 条可用；
  在可行分辨率下训练后全面变差。**计划中"用 GRPO 强化空/多块/短组"这一目标没有实现，
  但被 SFT 本身以更高水平达成了。**
- **唯一的剩余弱项是 multi_frag（85.6%）**，且 GRPO 也没能改善它。

---

## 8. 下一步（等你决定）

1. **直接开 SFT_Q2**（推荐）：底模用 `checkpoints/q1_2b_merged`，数据
   `data/splits_stage_q2_ocr_mt_v2`，命令见 `docs/PIPELINE_QWEN35_2B.md` §7。
2. 若仍想要 Q1 的 RL：需要 **≥8MP 可用的 24GB 级 GPU**（本地/WSL 8GB 做不到），
   且要先把 signal 样本凑到 ≥500（当前只有 ~110）。
3. 可选：把 multi_frag 作为专门切片，补数据或单独小规模 SFT 强化。

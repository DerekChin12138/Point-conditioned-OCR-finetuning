# GRPO_Q1 HQ-200：信号优先的 GRPO 数据 + 训练链路

日期：2026-09-28。目标：**再试一次 GRPO**，但这次先把「为什么上次失败了」解决掉。

上一轮（`docs/GRPO_Q1_V2.md` §6）失败三因：

| 上次 | 本次对策 |
|---|---|
| 探针 503 条只有 25% 有学习信号，实取 110 条 | 候选池 1500，**信号门禁前置**（`0.125≤frac_good≤0.875` 且 `std≥0.03`），目标产出率 ≥40% |
| 被迫 **1440²/4096**（SFT 是 2880²/8192）→ 分布外优化 | **强制与 SFT 同分辨率**（2880² / seq 12288） |
| 数据 110 条过少、且 40% 是已饱和的 empty | 200 条，**只打弱点桶**，empty 配额 = 0 |

---

## 1. 文件

| 文件 | 作用 |
|---|---|
| `data/scripts/build_grpo_candidates.py` | 建**候选池**（1500 train / 120 val，train-only）|
| `eval/run_grpo_value_probe.py` | G=8 rollout 探针（复用现有） |
| `data/scripts/compose_grpo_q1_hq200.py` | 门禁组数据 → **200 train / 60 val** |
| **`train/run_grpo_hq.sh`** | **通用一条龙**（Q1/Q2 × 0.8B/2B/4B）：status/cand/seed/probe/compose/dryrun/train/merge/eval/all |
| `train/run_grpo_q1_hq200.sh` | 旧名包装（= `KIND=q1`） |
| `tests/test_grpo_hq200.py` | 10 条单测（分类/去重/配额/门禁） |
| 产物 | `data/splits_grpo_cand/`、`data/splits_grpo_q1_hq200/` |

数据可由脚本**确定性重建**（`--seed 42`，输入是已有的 `data/splits_stage_q1_withreal/`），
所以上云不需要传数据文件本身。

---

## 2. 数据设计（为什么它「超高质量」）

### 2.1 只打实测弱点（reason 配额）

| reason | 最终 200 条 | 依据 |
|---|---|---|
| `multi_frag` | **77**（38%） | Q1 0.8B 0.691 → 2B 0.859，最大缺口 |
| `real` | **50**（25%） | 真实截图 Q1 0.842 / Q2 0.500，最弱桶 |
| `semantic_group` | 30（15%） | 短块组边界 |
| `long_block` | 25（12.5%） | 长块（复读/截断风险） |
| `repeat_risk` | 18（9%） | ♫/重复行/表格，解码退化风险 |
| `empty` | **0** | 已饱和（99.1%），喂它 = 0 梯度 |

### 2.2 多样性与尺寸鲁棒性
- **每页 ≤2 条**、模板按公平份额、marker 面积三分位均衡（实测 75/57/68）→ 200 条覆盖 ~100 张不同截图；
- dHash（hamming ≤4）去近重复 crop；
- 全部为正例（`is_negative=False`），GT 非空。

### 2.3 信号门禁（compose 的核心）
探针每条 G=8 个 rollout，用**与训练完全相同的 reward**（q1v2 / q2，且会读 `GRPO_REWARD_WEIGHTS`）打分：

| 判据 | 默认 | 含义 |
|---|---|---|
| `reward_max_frac ≥ min_reward_max_frac` | **0.95** | 组内**必须有一条近乎满分**的轨迹可强化 |
| `reward_std ≥ min_std` | **0.10** | 组内要有**足够方差**（有高有低才有 advantage） |
| `min_good_frac ≤ frac_good ≤ max_good_frac` | 0.125 / 0.875 | 不能全饱和、也不能全崩 |
| 排序 | `reward_max_frac` → `reward_std` → 失败模式价值 | 先挑"最好那条很好、且分布很散"的组 |

失败模式价值 = `frac_over + frac_under + 0.5×frac_empty_miss`（Q1）/ `frac_over + 0.5×frac_miss + 0.5×frac_badxml`（Q2）。
「有时对、有时错」才是 GRPO 能学的样本；全对/全错 advantage=0，直接丢弃。

调参入口：`--min-reward-max-frac`、`--min-std`、`--good-frac`、`--min/max-good-frac`（compose 阶段）。

---

## 2.5 Q2 版本（定位优先的 RL）

4B 走完 Q2 后，为了让「点 OCR 近乎完美再蒸馏」，用 **Q2 格式**的 HQ 数据在同一套链路里做 RL：

```bash
KIND=q2 SIZE=4b GEN_BATCH=4 bash train/run_grpo_hq.sh probe     # 数据 = data/splits_grpo_cand_q2/
KIND=q2 SIZE=4b bash train/run_grpo_hq.sh compose               # → data/splits_grpo_q2_hq200/（200/60）
KIND=q2 SIZE=4b CONFIRM=1 bash train/run_grpo_hq.sh all
```

Q2 默认用**定位优先权重**（翻译几乎不计分，`source,translation,xml,empty,halluc,over,leak`）：

```
GRPO_REWARD_WEIGHTS="1.5,0.05,1.0,1.5,1.5,0.75,0.5"
```
- 该变量被 **probe / compose / 训练** 三处共同读取（`weights_from_env()`），保证门禁与训练同尺度；
- `positive_ceiling = source+translation+xml = 2.55`，gate 用 `frac_good ≥ 0.8 × ceiling`；
- 想恢复默认（翻译权重 1.0）就不设这个变量。

## 3. 运行

```bash
cd ~/autodl-tmp/ocr-mt-finetuning
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 0) 先无 GPU 出一份 seed（signal 未验证，用来冒烟链路）
bash train/run_grpo_q1_hq200.sh seed

# 1) 候选池（5 秒，确定性）
bash train/run_grpo_q1_hq200.sh cand

# 2) 探针（需要 GPU；这是最贵的一步）
SOURCE scripts/setup_autodl_mirrors.sh 2>/dev/null || true
ADAPTER=checkpoints/q1_08b/adapter_final bash train/run_grpo_q1_hq200.sh probe
#   显存够就加大生成批：GEN_BATCH=4 ... probe

# 3) 门禁组数据 → 200/60
bash train/run_grpo_q1_hq200.sh compose

# 4) 冒烟（16 样本 / 2 步）→ 看显存
bash train/run_grpo_q1_hq200.sh dryrun

# 5) 正式训练
CONFIRM=1 ADAPTER=checkpoints/q1_08b/adapter_final bash train/run_grpo_q1_hq200.sh train

# 6) 读板对比
bash train/run_grpo_q1_hq200.sh eval
```

**探针成本**（4080S 估算）：1500 条 × G=8 = 12000 次 rollout @2880²。
`GEN_BATCH=1` 约 6-8s/条 → **20h+**；把 `GEN_BATCH` 提到 4-8（32GB 装得下：0.8B + 4×8500 tok KV ≈ 2GB）
可降到 **4-6h**。想更快就先只探 `--max-samples`（改 `run_grpo_value_probe.py` 的 `--max-samples`）。

---

## 4. 训练参数（`train/run_grpo_q1_hq200.sh train` 已内置）

| 项 | 值 | 理由 |
|---|---|---|
| `--model` / `--adapter` | `models/Qwen3.5-0.8B` + `checkpoints/q1_08b/adapter_final` | `--adapter` 不加 `--continue-adapter` = 烘成**冻结 π_ref** + 挂新 LoRA |
| LoRA | **r16 / α32**（比 SFT 的 r32 小） | RL 只微调策略，不需要大容量 |
| `--lr 5e-6` / `--beta 0.04` | 小步 + 弱 KL 约束 | 防止破坏 SFT |
| `--num-generations 8` `--temperature 1.0` | G=8 | 组内 advantage |
| `--gen-chunk-size 1` | 生成分块 | 8GB 友好；32GB 可升到 2-4 |
| `--max-completion-length 256` | 块级任务够用 | 长块溢出由 edit/under 惩罚 |
| `--batch-size 2 --grad-accum 8` | 有效 16 | 与 SFT 一致 |
| `--max-seq-length 12288 --max-pixels 8294400` | **与 SFT 完全一致** | 不降分辨率 |
| `--save-steps 25` | 频繁存 | 200 条 1 epoch 很短 |
| `--min-pixels 200704` / `--collator-resize` | 不适用（GRPO 走 generate） | — |

---

## 5. 门禁（先定好，避免又一次无效长跑）

**投入前**
- 探针产出率 = `compose` 里 `probe_rows → n_train(200)` 的转化率。**< 40% 就不做**（说明 0.8B 已饱和）。

**训练中**
- 前 50 步 `reward` 不上升 → 停（`train/unsloth_grpo.py` 日志/TensorBoard）。

**训练后**（同一 val、同一 `--gen-batch-size`、同一分辨率对比）
| 指标 | 成功线 |
|---|---|
| `block_hit_rate`（Q1 val） | 整体**不下降**（±0.3pp 内） |
| `multi_frag` / `semantic_group` / `real_labeled` 分桶 | **各 +2pp** |
| `empty_on_chrome_rate` | ≥ 0.97 |
| `format_leak_rate` | = 0 |
| 未达标 | **判定失败，回退 SFT 权重**（不要拿 GRPO 权重部署） |

---

## 6. 结果记录

| 日期 | 配置 | 探针产出 | val block_hit | mf | sg | real | 判定 |
|---|---|---|---|---|---|---|---|
| 2026-09-24 | 2B / 1440² / 110 条 | 25% | 下降 | − | − | − | ❌ 丢弃 |
| 待填 | 0.8B / 2880² / 200 条 | ? | ? | ? | ? | ? | ? |

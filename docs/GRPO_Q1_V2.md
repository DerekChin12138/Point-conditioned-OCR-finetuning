# GRPO_Q1 v2 — 单轮、~500 条、强化空/多块/短组

日期：2026-09-23。目标：用**一轮** GRPO 综合强化 **空标记 / 多块（多 fragment 整段）/ 分点短文本块组**，
并让标记尺寸不再是 GRPO 的学习任务（尺寸变化改由 SFT 数据承担，见 [`MARKER_SPEC.md`](MARKER_SPEC.md)）。

相关文件：

| 角色 | 文件 |
|---|---|
| 数据配比 | `data/recipes/grpo_q1_v2.yaml` |
| 探针（筛有学习价值样本） | `eval/run_grpo_value_probe.py --reward-set q1v2` |
| 筛选 + 组数据 | `data/scripts/compose_grpo_q1_v2.py` |
| 奖励函数 | `src/point_ocr/grpo_rewards_q1.py` |
| 训练 | `train/unsloth_grpo.py --reward-set q1v2` |
| 回归测试 | `tests/test_grpo_rewards_q1.py` |

---

## 1. 为什么这样筛（依据 SFT_Q1 结果）

SFT_Q1 在 regular val 上：hit **87.6%**、over-extraction 0.56%、empty-on-chrome 98.7%。
也就是说**普通块和空标记在常规页上已经好**，剩下的误差集中在难例：

| 切片 | SFT_Q1 source hit |
|---|---|
| core_inner（普通长块） | ~95% |
| multi_frag（跨 fragment 整段） | ~75% |
| semantic_group（短块组 / 分点） | ~77% |

而且 Q2 的诊断证明：**空输出纪律一旦离开训练分布就会崩**（Q2 负例只有 54% 空）。
所以 v2 的数据和奖励都把重心压到这三块。

## 2. 数据标准（约 500 条）

配比（`grpo_q1_v2.yaml`）：

| 场景 | 占比 | 条数 | 目的 |
|---|---|---|---|
| `empty`（负例：空白/表格/chrome/交界） | 40% | 200 | **空标记** |
| `multi_frag`（跨 fragment 整段） | 25% | 125 | **多块** |
| `semantic_group`（紧邻短块组/分点） | 25% | 125 | **分点短文本块组** |
| `regular`（core_inner 普通块） | 10% | 50 | 防回退 |

空例内部再分：`empty_clear 55% / empty_special 35% / empty_boundary 10%`，chrome 占 50%。

**"有学习价值"的判定**（对每条样本跑 G=8 rollout，用**与训练相同的 q1v2 reward** 打分）：

> ❗ 旧的 `max R > 0.5` 不能再用了：v2 reward 的**量程按场景不同**。
> 正例 ceiling = `edit` = **1.0**（empty/over/under 只会扣）；
> 负例 ceiling = `edit + empty` = **1.0 + 1.5 = 2.5**（空↔空）。
> 固定的 0.5 在正例上=“半对”，在负例上=“四分之一条空轨迹”，语义不一致。

改为**按场景 ceiling 归一** + **按失败模式排序**：

1. `good` = reward ≥ `good_frac`（0.8）× 该场景 ceiling；
2. **硬门**：`min_good_frac ≤ frac_good ≤ max_good_frac`（0.125–0.875）**且** `reward_std ≥ min_std`（0.03）
   —— 至少一条好轨迹（有可强化的路径）+ 至少一条差轨迹（有方差才有 advantage）；
   全饱和 / 全崩（std=0）都不要；
3. **价值排序**（在每个场景配额内取 top-N）：
   - 正例：`frac_over + frac_under + 0.5 × frac_empty_miss`
     —— 优先要**真倾倒了**或**真缺成员**的组，正好是 multi_frag / 短组边界；
   - 负例：`1 - 2 × |frac_good - 0.5|`
     —— 优先要**空/幻觉各半**的组。

> 与旧管线最大的不同：旧 GRPO_1/GRPO_2 的 train **一条负例都没有**，TB 里
> `empty_reward/mean ≡ 0`，空轴完全没被 RL 碰过。v2 强制 40% 负例。

Val 80 条，均匀 regular / multi_frag / semantic_group / empty（来自 SFT val，不参与训练）。

## 3. 奖励函数（`grpo_rewards_q1.py`）

| 项 | 权重 | 作用 |
|---|---|---|
| `edit_reward` | 1.0 | 块级 edit 相似度（保住定位） |
| `empty_reward` | **1.5** | 空 GT 出空 = +1；正例出空 = −1 |
| `over_extraction_penalty` | 1.0 | 正例倾倒多块/整页 = −1（**多块**） |
| `under_extraction_penalty` | 1.0 | 正例只出一部分（sim<0.55 且长度<0.65×GT）= −1（**分点短组缺成员**） |
| `leak_penalty` | 0.5 | chat/think 控制符泄漏 = −1 |

设计理由：

- **空权重最高**：这是最容易在换分布时崩的轴，也是最容易被错误奖励毁掉的轴。
- **过抽/欠抽分开计**，不并进 hit：这样 G=8 的组内还能把"缺一个成员"和"倾倒四块"分出名次，
  正好对应 multi_frag / semantic_group 的边界问题。
- **不给尺寸奖励**：尺寸变化已在 SFT 数据里（每图 0.08%–0.2% 均匀随机），GRPO 不该再花容量学它。
- 惩罚项只在正例生效，负例完全由 `empty_reward` 决定，避免双重惩罚把空例推乱。

## 4. 运行顺序（开训由你执行）

```bash
# 前提：SFT_Q1 已在新标记（0.08%-0.2% 随机）数据上重训，得到 adapter

# 1) 探针：G=8，覆盖 empty/regular，用 q1v2 reward 打分
uv run python eval/run_grpo_value_probe.py \
  --src data/splits_stage_q1_withreal/train.jsonl \
  --out checkpoints/grpo_value_probe_v2 \
  --adapter checkpoints/<q1_v2>/adapter_final \
  --model models/Qwen3.5-2B \
  --reward-set q1v2 --scenes regular,multi_frag,semantic_group --include-empty \
  --g 8 --temperature 1.2 --max-new-tokens 256

# 2) 组 500 条
uv run python data/scripts/compose_grpo_q1_v2.py

# 3) 单轮 GRPO
uv run python train/unsloth_grpo.py \
  --data data/splits_grpo_q1_v2/train.jsonl \
  --val  data/splits_grpo_q1_v2/val.jsonl \
  --model models/Qwen3.5-2B \
  --adapter checkpoints/<q1_v2>/adapter_final --continue-adapter \
  --run-id grpo_q1_v2 --reward-set q1v2 \
  --num-generations 8 --temperature 1.0 --top-p 1.0 --beta 0.04 \
  --max-completion-length 256 \
  --lr 5e-6 --epochs 1.0 --lora-rank 16 --lora-alpha 32 \
  --gen-chunk-size 2 --save-steps 50 --logging-steps 5 \
  --max-grad-norm 1.0 --optim adamw_8bit --num-workers 4
```

> 2B 上 G=8 × 大图 generate 是显存峰值场景，先 `--gen-chunk-size 1`（或 `--num-generations 4`）跑通再放大。

## 5. 本机性能（已加到两个训练脚本）

| 项 | 值 |
|---|---|
| CPU / RAM / shm | 32 线程（i9-13980HX）/ 15GB / `/dev/shm` 7.8GB |
| DataLoader | `--num-workers 8`（SFT 默认）、`--persistent-workers`、`--prefetch-factor 4`、`--pin-memory`（默认开） |
| TF32 | `--tf32`（默认开） |
| 显存分配器 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（脚本已设默认） |
| 建议 batch | 2B 4bit 8.3MP+8192：`2×8`（峰值 ~3.9GB）；还有 ~4GB 余量，可试 `4×4` |

关掉某项：`--no-pin-memory` / `--no-persistent-workers` / `--no-tf32`。

---

## 6. 实测结论（2026-09-24）：**本轮 GRPO 不可行，丢弃**

Q1 SFT（2B）训完后按本文流程跑了一遍，结果如下（同一 500 条分层 val、全分辨率）：

| 指标 | SFT_Q1 | SFT_Q1 + GRPO_Q1v2 | Δ |
|---|---|---|---|
| `block_hit_rate` | **0.9396** | 0.9154 | **−2.4pp** |
| `empty_on_chrome` | 0.9882 | 0.9763 | −1.2pp |
| `mean_edit_similarity` | 0.9652 | 0.9518 | −0.013 |
| regular | 0.971 | 0.964 | −0.7pp |
| multi_frag | 0.856 | 0.844 | −1.2pp |
| semantic_group | 0.971 | 0.913 | **−5.8pp** |

**GRPO 全面变差 → 不采用，Q1 里程碑保留 SFT_Q1（`checkpoints/q1_2b_merged`）。**

三个原因（按重要性）：

1. **信号不足**：探针 503 条只有 127 条 signal（25%）；`empty` 几乎全饱和
   （reward 恒 2.5，std=0），最终只有 110 条可训练（mf 62 / sg 36 / reg 8 / empty 4），
   且训练时 `frac_reward_zero_std = 0.4` —— 40% 的组没有梯度。
   → 2B SFT 已经把目标能力基本解决了，RL 没什么可学。
2. **分辨率被迫降级**：8GB + WSL 跑不动 8.3MP × G≥4 的 GRPO
   （`dxgkio_make_resident: Ioctl failed -12` → `CUDA driver error: device not ready`），
   实际用 **1440²/4096** 训练，而 SFT 是 2880²/8192 → 分布外优化，越训越偏
   （train reward 0.907 → 0.860）。
3. 小数据 + 低 lr 也不足以避免对 semantic_group 的破坏（−5.8pp）。

**后续若要再试 GRPO，先满足**：≥ 8MP 的 24GB 级 GPU（或云端）、
signal 样本 ≥ 500、且 `empty` 切片有非零方差。否则 SFT 已经够用。

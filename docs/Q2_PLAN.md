# Q2 (点 OCR + 英译中) 后续计划

日期：2026-09-23。基线：`checkpoints/q2_ocr_mt/adapter_final`（Q1 GRPO merge 上挂 Q2 SFT LoRA）。
评测口径：`eval/run_q2_eval.py`（`src/point_ocr/q2_metrics.py`）。

---

## 0. Q2 SFT v1 复盘（一句话）

**点定位没坏（source hit 89.3%），翻译大体可用，但空输出纪律崩了（有效空率 65.8%，34.2% 在负例上幻觉整块）。**
根因不是准星尺寸、也不是 Q2 SFT 破坏了 Q1——对照实验证明是 **Q2 负例数据分布与 Q1 不同**（prose 表格 + 落在 chrome/墨迹上的 `neg_clear`）。

| 指标 | Q2 SFT v1 | 说明 |
|---|---|---|
| source hit (≥0.85) | **89.3%** | 与 Q1 的 87.6% 持平 → 没回退 |
| source edit-sim | 0.931 | |
| translation chrF2 | 0.341 | 参考译文是 OPUS 唯一措辞，合理意译也低 |
| both hit (source∧translation) | 10.8% | |
| XML well-formed | 98.4% | 1.5% 没闭合 `</translation>`（256 token 上限） |
| strict empty | 54.1% | |
| **effective empty** | **65.8%** | 把空壳/裸 `<source>` 也算空 |
| hallucination（负例出非空 source） | **34.2%** | 重灾区 |
| plain text（负例无 XML） | 9.1% | |
| format leak | 0 | |

对照实验（同一 `q1_grpo_merged`）：

| 集合 | 有效空率 |
|---|---|
| Q1 自身负例（80，含 33 special_table） | 97.5%（special_table 0/33 泄漏） |
| Q2 负例（100，含 42 special_table） | 50%（special_table 37/42 泄漏） |

控制变量（同模型、同 prompt、同准星 0.005/0.00425/0.0025）：**Q1 special_table 页 → 空；Q2 special_table 页 → 三种尺寸全部泄漏。**
→ 变量是**页面内容**：Q2 用 OPUS 把 `02_table_formula` / `06_dense_table_cells` 的单元格填成了整句/整段 prose，视觉上和 `core_inner` 要求输出的正文块几乎一样。

产物：`eval/results/q2_sft_analysis/`（含 3 个探针 JSON、对照脚本、抽查截图）。

---

## 1. 本轮已落地的工程改动

| 改动 | 文件 |
|---|---|
| Q2 专用指标（拆 source / translation / empty，chrF2 + chrF++） | `src/point_ocr/q2_metrics.py` |
| 离线打分脚本（无需 GPU，读已有 predictions） | `eval/run_q2_eval.py` |
| 负例清洗规则（prose 表格 / 落在墨迹的 clear / chrome 错标） | `src/point_ocr/q2_filter.py` |
| 负例清洗入口（splits / pools 两模式 + 纯空负例配平） | `data/scripts/filter_q2_negatives.py` |
| 清洗后数据 | `data/splits_stage_q2_ocr_mt_v2/` |
| 推理 token 上限 256 → **512** | `src/point_ocr/infer.py` |
| Q2 GRPO reward 雏形（7 项） | `src/point_ocr/grpo_rewards_q2.py` |
| GRPO 脚本支持 `--reward-set {q1,q2}` | `train/unsloth_grpo.py` |
| Studio Q2 启动器 | `eval/marker_studio/run_q2.sh` |
| `--finetune-vision` 真正生效（Qwen3.5 上原本是 no-op，见 §2.6） | `train/unsloth_stage_a.py` |

### v2 数据做了什么

`train: 49748 → 49655`（neg 11250 → 11157，比例 22.5% 保持）

| 规则 | train 删除 | 理由 |
|---|---|---|
| `special_prose` | 1057 | 单元格是 prose，与 core_inner 正例矛盾 |
| `clear_on_ink` | 588 | `neg_clear` 点落在文字块内，GT 空是错标 |
| `chrome_mislabel` | 40 | 标了 `chrome_*` 但点不在 chrome 带内 |
| `neg_clear`→`chrome_head/side` 重标 | 1074 | 点落在 chrome 带（标签错、训练信号本身没错） |
| 纯空负例上采样 ×1.35 | +1592 | 补回负例比例（val/test 不上采样） |

val/test 只删不采样：val 2764→2675（neg 536），test 2763→2663（neg 525）。

---

## 2. 指标定义与门禁

`eval/run_q2_eval.py` 输出的每个数字：

| 指标 | 计算 | 反映 |
|---|---|---|
| `xml_pair_rate` | 正例中解析出 `<source>` 的比例 | 是否会输出结构 |
| `xml_well_formed_rate` | 双标签齐全 + 顺序对 + 标签外无游离文本 | 结构稳定性 |
| `source_hit_rate` | `edit_similarity(pred_source, gt_source) ≥ 0.85` 的比例 | **点定位/OCR 能力** |
| `source_edit_similarity` | 同上相似度的均值 | 定位的连续质量 |
| `translation_chrf` / `chrfpp` | 解析出的译文 vs 参考译文的 chrF2 / chrF++ | **翻译质量（比 edit 公平）** |
| `translation_hit_rate` | chrF2 ≥ 0.60 | 翻译命中率 |
| `both_hit_rate` | source hit 且 translation hit | 端到端可用率 |
| `translation_cjk_ratio` | 译文中 CJK 字符占比 | 是否真的在翻译（不是照抄英文） |
| `translation_length_ratio` | len(pred_zh)/len(gt_zh) | 漏译 / 啰嗦 |
| `unclosed_translation_rate` | 有 `<translation>` 无 `</translation>` | 截断 / 复读 |
| `positive_over_extraction_rate` | source 明显长于 GT | 整页倾倒 |
| `strict_empty_rate` | 负例预测严格等于 `""` | 空输出纪律 |
| `effective_empty_rate` | 空串 + 空壳 + 裸 `<source>` | 空输出纪律（含格式坏的） |
| `broken_empty_rate` | 空壳/裸 `<source>`（有效空但不干净） | 格式 vs 幻觉的分界 |
| `hallucination_rate` | 负例出现非空 source | **幻觉率** |
| `plain_text_rate` | 负例无任何 XML 标签 | 格式回退 |
| `format_leak_rate` | 原始解码含 chat/think 控制符 | 解码纪律 |

### 门禁（Q2 毕业线）

| 指标 | 目标 |
|---|---|
| `source_hit_rate` | ≥ **0.88**（不得回退） |
| `effective_empty_rate` | ≥ **0.85** |
| `hallucination_rate` | ≤ **0.10** |
| `broken_empty_rate` | ≤ **0.05** |
| `xml_well_formed_rate` | ≥ **0.97** |
| `unclosed_translation_rate` | ≤ **0.01** |
| `translation_chrf` | ≥ **0.34**（不退化；chrF 是代理，别当唯一目标） |
| `both_hit_rate` | ≥ **0.20** |
| `format_leak_rate` | = 0 |
| Studio 目视 | XML 成对、空例仍空、Q1 跟点不回退、译文可读 |

---

## 2.5 评测加速（post-eval 7h → ~1.8h）

**问题**：post-eval 是 batch-1 greedy，2764 条要 ~7h（~9.6s/条，GPU 利用率 ~24%，decode 延迟受限）。

**改动**：
- `src/point_ocr/infer.py::generate_point_batch`：批量 generate（VL processor 左 padding），按图片面积排序减少 padding，`stop_strings`/EOS 行为不变。
- `train/run_observability.run_final_generate_eval`：分块流式加载（只驻留 ~2 个 chunk，不再把整集图片读进内存）+ CPU 预取线程与 GPU 生成重叠 + OOM 自动二分回退。
- 新参数：SFT `--post-eval-batch-size`（默认 8）、adapter eval `--gen-batch-size`（默认 8）。

**实测**（RTX 4060 8GB，4-bit，Q2 val）：

| 配置 | samp/s | 相对 batch1 | 2764 条预计 |
|---|---|---|---|
| batch1（旧） | 0.10–0.11 | 1.0× | ~7h |
| **batch8（新默认）** | **0.42–0.45** | **~4×** | **~1.8h** |
| batch16 | 0.13 | 0.3× | 更慢（大图 padding / 显存抖动） |
| bf16 batch8 | 0.27 | 2.3× | 不如 4-bit |

结论：**4-bit + batch8** 最优；bf16 反而更慢（小模型 decode 受显存带宽限制，4-bit 权重更小）。

**精度影响**（同一 48 条，batch1 vs batch8）：prediction 完全相同 26/48，**source 相同 46/48，空/非空判断 48/48 一致**；`strict/effective_empty`、`hallucination`、`format_leak` 三项**完全不变**，`source_hit` ±0.03（≈1 条），chrF ±0.001。批量 greedy 与 batch-1 不是逐 bit 一致（浮点归约顺序），但**结构性指标不漂移**。要求：所有模型对比都用同一 batch size（默认 8）。

证据：`eval/results/q2_sft_analysis/eval_batch_drift.txt`。

---

## 2.6 `--finetune-vision` 真相 + GRPO 复盘

### `--finetune-vision` 之前是 no-op（已修）

实测（真实加载 Qwen3.5-0.8B 建 LoRA）：`finetune_vision_layers=True/False` **结果完全一致**。`target_modules="all-linear"` 总会给 vision tower (`model.visual.*`) 挂 LoRA，且始终可训练：

| 设置 | total trainable | vision LoRA 可训练 |
|---|---|---|
| `finetune_vision_layers=True` | 26,363,904 | 96 tensors / 4.72M |
| `finetune_vision_layers=False` | 26,363,904 | 96 tensors / 4.72M |

后果（检查已保存 adapter 的 `visual.*.lora_B` 是否非零）：

| run | 配置 | vision 实际是否训练 |
|---|---|---|
| Q1 SFT | `--finetune-vision` | ✅ 48/48 非零 |
| GRPO_1 / GRPO_2 | `--adapter` 续训（冻结逻辑生效） | ❌ 0/48（保持 0） |
| **Q2 SFT v1** | 未传 flag，本意冻结 | ❌❌ **实际训练了 vision**（48/48 非零） |

**修复**：新增 `_freeze_vision_lora()`，对 fresh LoRA 也生效（`train/unsloth_stage_a.py`）。修后实测：flag=False → vision 可训练 0，总数 26.36M → 21.65M；flag=True 不变。回归测试 `tests/test_vision_freeze.py`。

> ⚠️ **Q2 v1 的结果是在「vision 被训练」下得到的**。v2 若沿用 v1 命令，现在会真正冻结 vision —— 行为变了，是有意修复。想复现 v1 就显式加 `--finetune-vision`。

### GRPO 两轮复盘 → 建议并成一轮

历史：`grpo_q1_hard`（GRPO_0，750 条含 225 负例，被作废）→ `grpo_q1_signal`（GRPO_1）→ `grpo_q2_marker`（GRPO_2，最终 merge 进里程碑）。

两轮**方法完全相同**（同 reward/权重、lr 5e-6、β 0.04、r16/α32、G=8、T=1.0、π_ref = SFT merge、同场景），只差数据：

| | GRPO_0 | GRPO_1 | GRPO_2 |
|---|---|---|---|
| train n | 750 | 491 | 200 |
| **负例** | **225** | **0** | **0** |
| 标记尺寸 | x45c | x45c | 混 ×0.5/×0.85 |
| 场景 | mf 55 / sg 40 / empty 5 | mf 323 / sg 106 / real 61 | mf 149 / sg 25 / real 26 |
| val hit | 0.717 | 0.783 | 0.817* |
| val empty | 1.0 | 1.0 | **0.95** |

TB 证据：两轮的 `empty_reward/mean` 与 `leak_penalty/mean` **全程恒为 0** —— train 里没有负例，空例奖励根本没有梯度，GRPO 只优化了 edit。GRPO_1 edit 0.644→0.827；GRPO_2 0.828→0.781（数据更难，标记变小）。

**结论：并成一轮。** 理由：

1. 两轮没有方法论差异，只是数据分批 → 单轮合并数据集等价。
2. GRPO_2 唯一的独特点（混标尺寸）本就该放进单轮数据。
3. 两轮 train 都无负例 → 空轴完全没被 RL 修，GRPO_2 反而在 val 上掉 5pp。
4. 少一次 compose + probe + run + merge，链路更短、更好复现。

仅当第二轮是**真正不同的分布**（如新 OOD 真实图）或明确课程（易→难）时才值得两轮；现在不是。

---

## 3. 训练计划

### P0 — Q2 SFT v2（数据唯一变量，先跑这个）

用 v2 数据、**与 v1 完全相同的超参**，隔离"数据修复"的效果。

```bash
uv run python train/unsloth_stage_a.py \
  --data data/splits_stage_q2_ocr_mt_v2/train.jsonl \
  --val  data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --model checkpoints/q1_grpo_merged \
  --out-root checkpoints --run-id q2_ocr_mt_v2 \
  --max-seq-length 4096 --batch-size 2 --grad-accum 8 \
  --lr 2e-5 --epochs 1.0 --lora-rank 32 --lora-alpha 64 \
  --save-steps 500 --eval-steps 250 --eval-batch-size 2 --num-workers 4 \
  --seed 42 --warmup-ratio 0.03 --weight-decay 0.01 \
  --lr-scheduler cosine --optim adamw_8bit --max-grad-norm 1.0
```

训完自动跑 post-eval（v1 时 post-eval 用了 ~7h；先冒烟再全量）。然后：

```bash
uv run python eval/run_q2_eval.py \
  --predictions checkpoints/q2_ocr_mt_v2/metrics/final_predictions.jsonl \
  --split data/splits_stage_q2_ocr_mt_v2/val.jsonl \
  --out-dir checkpoints/q2_ocr_mt_v2/metrics --prefix q2
```

**决策**：`effective_empty ≥ 0.80` 且 `source_hit` 不回退 → 进 P2；否则先做 P1。

### P1 — 若空例仍弱（三选一，别同时改）

1. **加空例权重**：把 `--blank-factor` 提到 1.6–2.0 重出 v2 数据，或对空例做 loss 加权（需在 `unsloth_stage_a.py` 加 sample weight）。
2. **换精度/视觉**：v1/v2 是 4bit QLoRA + 冻 vision；Q1 SFT 是 bf16 + vision LoRA。用 `--no-4bit --finetune-vision` 再跑一次对比（8GB 可行，Q1 已验证）。
3. **进 GRPO**（P2），让 RL 直接优化空输出。

### P2 — Q2 GRPO（reward 已就绪）

**单轮**（两轮已并成一轮，理由见 §2.6）。前提：Q2 SFT v2 达标；π_ref = merge 后的 Q2-v2 整模。**数据里必须有负例**（否则 `empty_reward` 无梯度）。

```bash
# 1) 先 merge 出 π_ref 底模（Q2 adapter 烘焙进 Q1 GRPO merge）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python export/merge_unsloth_lora.py \
  --base checkpoints/q1_grpo_merged \
  --adapter checkpoints/q2_ocr_mt_v2/adapter_final \
  --out checkpoints/q2_ocr_mt_v2_merged

# 2) 组 GRPO 数据（待写 compose 脚本，见下），然后：
uv run python train/unsloth_grpo.py \
  --data data/splits_grpo_q2_ocr_mt/train.jsonl \
  --val  data/splits_grpo_q2_ocr_mt/val.jsonl \
  --model checkpoints/q2_ocr_mt_v2_merged \
  --adapter checkpoints/q2_ocr_mt_v2/adapter_final --continue-adapter \
  --run-id grpo_q2_ocr_mt --reward-set q2 \
  --num-generations 8 --temperature 1.0 --top-p 1.0 --beta 0.04 \
  --max-completion-length 512 \
  --lr 5e-6 --epochs 1.0 --lora-rank 16 --lora-alpha 32 \
  --gen-chunk-size 2 --save-steps 50 --logging-steps 5 \
  --max-grad-norm 1.0 --optim adamw_8bit
```

GRPO 数据（待建 `data/scripts/compose_grpo_q2_ocr_mt.py`）建议场景配比：

| 场景 | 占比 | 目的 |
|---|---|---|
| `empty`（v2 清洗后的空白/表格/chrome 负例） | 40% | 主攻空输出 |
| `multi_frag` | 20% | 跟点回退 |
| `semantic_group` | 15% | 组边界 |
| `regular core_inner` | 15% | 防回退 |
| `real_labeled` | 10% | 真实分布 |

标记尺寸：**每图均匀随机 0.08%–0.2%**（新 protocol `x45r`，见 [`MARKER_SPEC.md`](MARKER_SPEC.md)）。尺寸鲁棒性由 SFT 数据承担，**不再作为 GRPO 的学习目标**。负例必须非零（上表 `empty` 40%），val 也要带负例。

reward 设计理由见 `src/point_ocr/grpo_rewards_q2.py` 文件头：**空例相关项权重最高（empty 1.5 + hallucination 1.5）**，source 只给 0.5（已经 89%，不需要 RL 再学），translation 用 chrF 不用 edit，xml 结构单独给分，over-extraction / leak 作为护栏。

### P3 — Studio 门禁 + 导出

```bash
bash eval/marker_studio/run_q2.sh        # → http://127.0.0.1:7865
```

门禁通过后再 merge / GGUF（见 `export/README.md`）。

---

## 4. Studio 使用（自行打标签看效果）

```bash
bash eval/marker_studio/run_q2.sh
# base    = checkpoints/q1_grpo_merged
# adapter = checkpoints/q2_ocr_mt/adapter_final（v2 训完改成 v2）
```

| 页面 | 地址 | 用途 |
|---|---|---|
| 探伤 | http://127.0.0.1:7865 | 放十字、改体积/解码/提示词，立刻看输出 |
| 标注 | http://127.0.0.1:7865/label | 真实截图画框、写块文案，导出到 `data/label_out/` |

要点：
- 提示词预设选 **`ocr_mt_v1`**（Q2）；切 `a2_v3` 可单独验证 Q1 跟点是否回退。
- 解码默认 greedy；打开采样（T=1.0, top_p=1.0, n=8）可看 GRPO 分布里还有没有正确轨迹。
- 内置真实截图库 `data/real_world_sample/`（14 张，另有 `nochrome/`），标注台可 `打开目录` 直接指向它。
- 首次点「生成」才加载模型；加载 base+adapter 约 10–20s。

---

## 5. 风险 / 回滚

- **不要覆盖** `checkpoints/q2_ocr_mt/`（v1 基线）；v2 用新 run-id。
- v2 数据是"删样例 + 上采样"，未重渲；如需更多干净表格负例，应重建 `empty_special` 池（换掉 OPUS prose 填充），而非继续删。
- chrF 仍是单参考代理；若后续要严格衡量翻译，接 COMET（需额外依赖）。
- GRPO reward 权重是雏形，开训前先用 `reward_breakdown()` 在一批 rollout 上核对分项分布，再调权重。

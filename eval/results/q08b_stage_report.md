# Qwen3.5-0.8B 两轮 SFT 阶段报告（对齐 2B 口径）

> ⚠️ **2026-09-28 补注（重要）**：本报告的所有数字都是在 **训练图像被 collator 静默压到 768px 宽**
> （`--max-pixels 8294400` 未生效）且**评测/Studio 用全分辨率**（train/eval 不一致）的条件下得到的。
> 因此 `source_hit 0.8303`、`real_labeled 50%` 等数字**系统性偏低**，不能当作 0.8B 上限。
> 修复见 [`README.md`](../../README.md) §六 “2026-09-28 修复”与 [`docs/PIPELINE_QWEN35_08B.md`](../../docs/PIPELINE_QWEN35_08B.md)。
> 分辨率修复后需要重跑基线。

日期：2026-09-28。执行：**SFT_Q1 → merge → SFT_Q2 → merge → Q2 post-train val**。
目标：用与 2B **完全相同的配方**重打 0.8B 基线，分离“容量差异”与“数据/协议差异”。

---

## 1. 配置

| 项 | SFT_Q1 | SFT_Q2 |
|---|---|---|
| 底模 | `models/Qwen3.5-0.8B` | `checkpoints/q1_08b_merged` |
| 数据 | `splits_stage_q1_withreal`（25562/1421/1418） | `splits_stage_q2_ocr_mt_v2`（49616/2684/2666） |
| 序列 / 精度 | 8192 / 4bit QLoRA r32·α64 | 8192 / 4bit QLoRA r32·α64 |
| lr / epochs | 5e-5 / 1.5 | 1.5e-5 / 1.0 |
| vision | `--finetune-vision`（训） | 冻结 |
| loader | `--num-workers 4 --no-pin-memory --no-persistent-workers` | 同 |
| 步数 / 时长 | 2397 步 ≈ 4.6h | 3101 步 ≈ 9.3h |
| best eval_loss | 0.0295 @1750（终点更低） | **0.1997 @3000**（仍在缓降） |

**内存事故与处置**：首轮用 `--num-workers 8 --pin-memory --persistent-workers`，宿主
`MemAvailable` 从 4.9GB 一路掉到 0.68GB（pin+persistent 在长跑里缓慢泄漏），被看门狗在
step 670 击杀保护 WSL；随后改为 **workers=4 / no-pin / no-persistent**，RAM 稳定在
~8GB used / ~7.7GB avail，从 `checkpoint-500` 续训完成。见 `scripts/supervise_08b.sh`（自愈）与
`scripts/watch_08b_training.sh`（看门狗）。

---

## 2. 结果

### 2.1 Q1（val 1421，`eval/run_merged_eval.py`-式 post-eval）

| 指标 | 0.8B 新 | 2B（`q1_2b`） | 0.8B 旧（污染） |
|---|---|---|---|
| `block_hit_rate` | **0.9034** | 0.9396 | 0.876 |
| `empty_on_chrome_rate` | 0.9712 | 0.9882 | 0.987 |
| `over_extraction_rate` | 0.0007 | 0.0020 | — |
| `mean_edit_similarity` | 0.9367 | 0.9652 | — |
| `format_leak_rate` | 0.0 | 0.0 | — |

→ Q1 只差 2B **3.6pp**，且远超旧（污染）基线。**0.8B 的点定位基本可用。**

### 2.2 Q2（val 2684 全量）—— **未过门禁**

| 指标 | 门禁 | **0.8B 新** | 2B（v2） | 0.8B 旧（v1） | 判定 |
|---|---|---|---|---|---|
| `source_hit_rate` | ≥0.88 | **0.8303** | 0.9140 | 0.8934 | ❌ |
| `source_edit_similarity` | — | 0.8748 | 0.9457 | 0.9309 | — |
| `translation_chrf` | ≥0.34 | **0.2978** | 0.3922 | 0.3407 | ❌ |
| `translation_hit_rate` | — | 0.0823 | 0.1791 | 0.1141 | — |
| `both_hit_rate` | ≥0.20 | **0.0767** | 0.1725 | 0.1085 | ❌ |
| `xml_pair_rate` | — | 0.9551 | 0.9925 | 0.9991 | — |
| `xml_well_formed_rate` | ≥0.97 | **0.9453** | 0.9883 | 0.9841 | ❌ |
| `effective_empty_rate` | ≥0.85 | **0.9908** | 0.9963 | 0.6576 | ✅ |
| `strict_empty_rate` | — | 0.9908 | 0.9963 | 0.5408 | — |
| `hallucination_rate` | ≤0.10 | **0.0092** | 0.0037 | 0.3424 | ✅ |
| `unclosed_translation_rate` | ≤0.01 | 0.0084 | 0.0014 | 0.0150 | ✅ |
| `plain_text_rate` | — | 0.0 | 0.0 | 0.0912 | ✅ |
| `format_leak_rate` | =0 | 0.0 | 0.0004 | 0.0 | ✅ |
| `empty translation` | — | 5.7% | 2.0% | 2.6% | — |

**门禁 8 项中 4 项未过：`source_hit`、`xml_well_formed`、`chrF`、`both_hit`。**

### 2.3 Q2 分桶（`source_hit` / chrF2）

| bucket | n | 0.8B 新 | 2B | Δ |
|---|---|---|---|---|
| core_inner | 1500 | 89.3% / 0.308 | 96.0% / 0.393 | −6.7pp |
| multi_frag | 375 | 69.3% / 0.274 | 80.0% / 0.384 | −10.7pp |
| semantic_group | 188 | 73.4% / 0.237 | 81.4% / 0.314 | −8.0pp |
| **real_labeled** | 76 | **50.0% / 0.367** | **81.6% / 0.605** | **−31.6pp** |
| empty_clear | 345 | eff-empty 98.8%（halluc 1.2%） | 99.4% | −0.6pp |
| empty_special | 200 | eff-empty 99.5%（halluc 0.5%） | 100.0% | −0.5pp |

---

## 3. 结论

1. **数据修复对 0.8B 同样有效**：`effective_empty 65.8% → 99.1%`、`hallucination 34.2% → 0.9%`。
   这证明 Q2 v1 的崩溃是数据问题，不是 0.8B 容量问题。
2. **0.8B 的 SFT 天花板确实存在，且在 Q2 上比 Q1 更明显**：
   - Q1 差 2B 3.6pp；
   - Q2 `source_hit` 差 **8.4pp**、`chrF` 差 0.094、`both_hit` 差 9.6pp；
   - 结构指标也退化（`xml_pair` 95.5% vs 99.3%，`empty translation` 5.7% vs 2.0%）。
3. **最刺眼的是 `real_labeled`：50.0% vs 2B 81.6%（−31.6pp）**。真实截图上的容量缺口远大于合成页。
   这是 0.8B 部署路线必须先解决的点。
4. **对比旧 0.8B 的“提升”要谨慎**：旧数字（source 89.3%）是在**未清洗负例 + 大准星(x45c/d) + seq4096 +
   已证伪 GRPO 底模 + vision bug** 下得到的，与本次 `source 83.0%` 不可直接比较；本次是干净口径。

---

## 4. 产物 / 复现

| 路径 | 说明 |
|---|---|
| `checkpoints/q1_08b/`、`checkpoints/q1_08b_merged/` | SFT_Q1 LoRA + Q1 整模（1.7G） |
| `checkpoints/q2_08b/`、**`checkpoints/q2_08b_merged/`** | SFT_Q2 LoRA + **可部署整模（1.7G）** |
| `checkpoints/q2_08b/eval_val/metrics/{final_report,q2_report}.{json,md}` | 本次全量 val 报告 |
| `checkpoints/q2_08b/eval_val/metrics/final_predictions.jsonl` | 2684 条逐条预测（可离线重打分） |
| `checkpoints/q1_08b/metrics/final_report.json` | Q1 val 报告 |
| `eval/results/qwen4b_feasibility/` | 4B 本地可行性实测 |

```bash
bash train/run_08b_pipeline.sh smoke            # 冒烟
CONFIRM=1 bash train/run_08b_pipeline.sh all    # 全流程（现默认 workers=4 / no-pin）
```

---

## 5. 评测基建问题（本次踩到）

- **post-eval 有 GPU 显存缓慢爬升**：bs=6 从 2.7GB 爬到 7875MiB（8188 上限），吞吐
  0.65 → 0.15 samp/s；全量 2684 条耗时 **~5.3h**（前 1.5h 快、后 3.8h 慢）。
- bs 标定（64 条）：bs2 0.204 / **bs4 0.348** / bs5 0.376 / bs6 0.376 / bs8 0.266 samp/s
  （bs8 峰值 7916MiB，最慢）。
- **建议**：长 eval 走**分片**（每片 ~500 条、独立进程），或 `--max-new-tokens` 收紧；
  不要把 bs 顶到 8。显存爬升是 `generate` 循环里的碎片/缓存累积，与批次大小无关，分片能重置。

---

## 6. 下一步（按你之前定的目标）

0.8B 现在**定位差 8pp、真实图差 32pp**，纯 SFT 已经到顶。可选：
1. **蒸馏**：用 2B/4B/API teacher 造“学生 on-policy 轨迹上的高质量目标”再训 0.8B（真正 OPD 需云卡）。
2. **两阶段**：0.8B 只做点 OCR（Q1 90.3% 已够），翻译交给纯文本 MT。
3. **数据补强**：真实截图占比从 2.8% 提高（尤其 `real_labeled` 的 50% 是短板）。

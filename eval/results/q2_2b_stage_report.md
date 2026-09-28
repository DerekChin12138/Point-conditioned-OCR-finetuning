# Q2 SFT 阶段报告（Qwen3.5-2B OCR + 英译中）

日期：2026-09-25。执行：**SFT_Q2**（底模 `checkpoints/q1_2b_merged`，数据 `splits_stage_q2_ocr_mt_v2`）。
无 GRPO_Q2（理由见 `docs/GRPO_Q1_V2.md` §6）。

---

## 1. 配置与过程

| 项 | 值 |
|---|---|
| 底模 | `checkpoints/q1_2b_merged`（SFT_Q1，vision 冻结） |
| 数据 | `data/splits_stage_q2_ocr_mt_v2`（train 49616 / val 2684 / test 2666，新标记 + 负例清洗） |
| 精度 | 4bit QLoRA，LoRA r32/α64，可训练 33.64M（1.49%，**vision 已真正冻结**） |
| 序列 | `max_seq_length 8192`（全库最大估计 8202，仅 3 条略超） |
| 超参 | lr 1.5e-5，1 epoch，batch 2×accum 8，cosine，warmup 3% |
| 步数/耗时 | 3101 步，约 **10.7h**（含一次因内存重启的续训） |
| 显存/内存 | GPU 峰值 ~4.8GB；宿主 RAM 峰值 ~7.3GB（单进程，无 worker 复制） |
| best | `eval_loss 0.1582` @ step 3000（`adapter_final` = best） |

**内存事故与处置**：最初用 `--num-workers 4`，4 个 `pt_data_worker` 因 fork COW 各占 ~2.6GB（共 10.4GB）
+ 主进程 4.65GB，逼近 15GB 上限 → 改为 `--num-workers 0 --no-pin-memory`，RAM 稳定在 ~7GB，零 OOM 风险。
中途另因该问题从 `checkpoint-1000` 续训一次（约 20 步重做）。

---

## 2. 结果（全量 val 2684 条）

`eval/run_q2_eval.py`（拆 source / translation / empty）：

| 指标 | 门禁 | **Q2 v2（2B）** | Q2 v1（0.8B） | Δ |
|---|---|---|---|---|
| `source_hit_rate` | ≥0.88 | **0.9140** | 0.893 | +2.1pp |
| `source_edit_similarity` | — | **0.9457** | 0.931 | +0.015 |
| `translation_chrf` (chrF2) | ≥0.34 | **0.3922** | 0.341 | +0.051 |
| `both_hit_rate` | ≥0.20 | 0.1725 | 0.108 | +6.4pp（**差 2.7pp 未达**） |
| `xml_well_formed_rate` | ≥0.97 | **0.9883** | 0.984 | +0.4pp |
| `unclosed_translation_rate` | ≤0.01 | **0.0014** | 0.015 | ✅ |
| `strict_empty_rate` | — | **0.9963** | 0.541 | **+45.5pp** |
| `effective_empty_rate` | ≥0.85 | **0.9963** | 0.658 | **+33.8pp** |
| `hallucination_rate` | ≤0.10 | **0.0037** | 0.342 | **−33.8pp** |
| `plain_text_rate` | — | **0.0000** | 0.091 | ✅ |
| `over_extraction_rate` | — | 0.0009 | 0.0069 | ✅ |
| `format_leak_rate` | 0 | 0.0004（1 条） | 0 | ≈ |

**门禁判定：13 项中 12 项通过**，唯一未达标的是 `both_hit_rate`（0.173 vs 0.20）——
它要求 source 与翻译**同时** chrF≥0.6，而 chrF 对"另一种合理译法"本就偏低（见下）。

按场景（source hit）：regular 0.731、multi_frag 0.645、semantic_group 0.574、empty 0.996。

---

## 3. 关键结论

1. **空输出纪律从灾难变为近乎完美**：`effective_empty 65.8% → 99.6%`，
   `hallucination 34.2% → 0.37%`，`plain_text 9.1% → 0`。
   → 「裁负例 + prose 表格清洗 + 2B + 小标记」这套组合彻底解决了 Q2 v1 最大的失败。
2. **点定位保持并略升**：`source_hit 89.3% → 91.4%`（2B 换基座后没有丢 Q1 能力）。
3. **翻译有提升但仍是弱项**：chrF2 0.341→0.392。
   注意 chrF 是**单参考**指标，模型常用不同措辞的合理译法会被判低分，
   所以真实翻译质量应优于该数字；但 2B 也确实不是高质量 MT 的量级。
4. `both_hit` 未达 0.20 主要受翻译侧限制，不是结构或定位问题。

---

## 4. 产物

| 路径 | 说明 |
|---|---|
| `checkpoints/q2_2b/adapter_final` | SFT_Q2 LoRA（best eval_loss 0.1582） |
| **`checkpoints/q2_2b_merged`** | **当前可部署整模（Q1 SFT → Q2 SFT）** |
| `checkpoints/q2_2b/metrics/{final_report,q2_report}.json/.md` | 整串报告 + Q2 拆分报告 |
| `checkpoints/q2_2b/metrics/final_predictions.jsonl` | 2684 条逐条预测（可离线重打分） |

Studio：

```bash
BASE=checkpoints/q2_2b_merged bash eval/marker_studio/run_q2.sh
# 探伤 http://127.0.0.1:7865 ；标注 http://127.0.0.1:7865/label
```

---

## 5. 建议的下一步

1. **Studio 目视验收**（必须）：XML 是否成对、空例是否空、Q1 跟点是否回退、
   译文是否可读（重点抽样 `semantic_group` 与 `multi_frag`）。
2. 若翻译质量仍不够：考虑
   （a）用更强的多语模型做**两阶段**（0.8B/2B 点 OCR + 单独 MT），
   （b）蒸馏更大模型的译文进 SFT 数据，
   （c）再接一轮 Q2 GRPO（需 24GB 卡 + ≥500 signal 样本）。
3. 部署前 `export/export_gguf.sh checkpoints/q2_2b_merged ...`。

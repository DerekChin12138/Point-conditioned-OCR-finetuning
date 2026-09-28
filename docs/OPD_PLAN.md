# OPD（on-policy distillation）落地调研与方案

日期：2026-09-28。目标：把 4B（最好先过 GRPO）的**点定位**能力蒸馏进 0.8B，翻译只求「能做」。
配套：`train/run_grpo_hq.sh`（GRPO 侧）、`docs/GRPO_Q1_HQ200.md`。

---

## 1. 结论速览

| 方案 | 现成程度 | 视觉支持 | 工作量 | 建议 |
|---|---|---|---|---|
| **A. TRL `GKDTrainer` + 薄 VL 适配层** | ✅ 框架现成（`lmbda=1` on-policy、`beta=1` reverse KL，正是论文引用的 GKD） | ❌ 需小改（2 个方法 + VL collator） | **~120 行** | ✅ **首选** |
| B. TRL `GKDTrainer` 原样用于**纯文本**阶段 | ✅ 零改动 | 不需要（无图） | 0 | ✅ 若走两阶段（翻译） |
| C. 自研 top-k reverse-KL 训练循环（论文原版） | ❌ | ✅ | ~200 行 | 备选（若 A 与 TRL 内部打架） |
| D. 序列级蒸馏（teacher 改写学生轨迹 → SFT） | ✅ 无需框架 | ✅ | ~80 行 | 最便宜，效果也最弱 |

**论文原文**：teacher = 4B+GRPO；学生 rollout → teacher 在学生 **top-k 支撑集**上给 log-probs → 学生做 **top-k reverse KL**（mode-seeking）。TRL 的实现是**全词表** reverse KL（`beta=1`），目标一致、只是没做 top-k 稀疏化。

---

## 2. 为什么 TRL `GKDTrainer` 基本能用（实测源码结论）

`trl==0.24.0`，`.venv/lib/python3.12/site-packages/trl/trainer/gkd_trainer.py`：

1. `GKDConfig` 有 `lmbda`（on-policy 概率）、`beta`（0=forward KL / 0.5=JSD / **1=reverse KL**）、`temperature`、`max_new_tokens`、`teacher_model_name_or_path`、`seq_kd`。
   → `lmbda=1.0, beta=1.0` 就是我们要的「纯 on-policy、mode-seeking」。
2. `compute_loss`（非 liger 路径）：
   ```python
   student_outputs = model(input_ids=..., attention_mask=...)          # ← 只传这两个
   teacher_outputs = self.teacher_model(input_ids=..., attention_mask=...)
   prompt_lengths = inputs["prompts"].shape[1]
   shifted_* = logits[:, prompt_lengths - 1 : -1, :]                   # ← 只取 completion 段
   loss = self.generalized_jsd_loss(..., beta=self.beta)
   ```
   - ❌ 不带 `pixel_values` → 视觉输入被丢（所以「Qwen3.5-4B→0.8B」不能直接用）。
   - ✅ **只对 completion 段算 logits**：`[B, ≤512, V=248320]`，bf16 ≈ 250MB → **32GB 毫无压力**（论文做 top-k 是为了省这个，我们用 TRL 全词表也不怕，因为序列只切在 completion 上）。
3. `generate_on_policy_outputs` 用 `inputs["prompts"]` 生成 → 同样只吃文本。
4. `training_step`：`if random() <= lmbda: 用学生重新生成` → on-policy。

**结论**：拦路虎只有「多模态 kwargs 没被透传」。改法很小：

```python
MM_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "mm_token_type_ids")

class VLGKDTrainer(GKDTrainer):
    @staticmethod
    def _mm(inputs):
        return {k: v for k, v in inputs.items() if k in MM_KEYS}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 与父类同构，仅把 **mm 透传给 student / teacher 两次前向
        ...

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        out = model.generate(input_ids=inputs["prompts"],
                             attention_mask=inputs.get("prompt_attention_mask"),
                             **VLGKDTrainer._mm(inputs),
                             generation_config=generation_config,
                             return_dict_in_generate=True)
        ...
```
再加一个 **VL collator**：复用 `UnslothVisionDataCollator` 产出的
`input_ids / attention_mask / labels / pixel_values / image_grid_thw`，另建 `prompts`（只 tokenize user 轮）。

---

## 2.5 已落地并通过本地自检（2026-09-28）

`train/unsloth_gkd_vl.py` 已实现上面的薄适配层（`VLGKDTrainer` + `VLDataCollator` + `perturb_lora` + eos 占位符 shim）。

本地 8GB 自检（448² 小分辨率、`--steps 2`）：

```bash
uv run python train/unsloth_gkd_vl.py --selfcheck --model models/Qwen3.5-0.8B \
  --data data/splits_grpo_q1_hq200/train.jsonl \
  --max-seq-length 768 --max-prompt-length 640 --max-pixels 200704 --min-pixels 200704 \
  --max-new-tokens 24 --steps 2 --batch-size 1
```
结果：
```
(a) teacher == student        → reverse KL = 0.000000        PASS   ← 切片/多模态对齐正确
(b) 扰动 LoRA 学生 vs 冻结基座 → loss = 1.322, grad_norm = 93.5 PASS   ← 学生路径连通、梯度流动
collator: prompt_len=299 completion=119 seq_len=759 pixels=(728,1536)
```

踩到的两个集成坑（已修，见文件内注释）：
1. TRL 0.24 + Unsloth 的 config 兼容层会把 GKD 专有字段（`lmbda/beta/max_new_tokens/...`）报成
   "not a valid SFTConfig argument"；实测字段仍在、可正常读取（打印确认 `lmbda=1.0 beta=1.0`）。
2. `SFTTrainer` 会把 `args.eos_token` 解析成占位符 `'<EOS_TOKEN>'`（不在 Qwen 词表）而报错 →
   `patch_eos_placeholder()` 把它映射到真实 eos id（幂等、只对该字面量生效）。

跑真蒸馏（teacher = 4B 的 Q2 GRPO 整模）：
```bash
uv run python train/unsloth_gkd_vl.py \
  --model checkpoints/q2_08b_merged            # 学生起点（0.8B Q2 SFT 整模）
  --teacher checkpoints/grpo_q2_4b_merged      # 教师（4B Q2 + GRPO）
  --data data/splits_grpo_q2_hq200/train.jsonl \
  --out checkpoints/opd_q2_08b \
  --max-seq-length 12288 --max-prompt-length 10240 --max-pixels 8294400 \
  --lmbda 1.0 --beta 1.0 --temperature 1.0 --max-new-tokens 256 \
  --mask-translation --lr 1e-6 --batch-size 1 --grad-accum 8
```
> 8GB 本地**跑不了真蒸馏**（teacher 4B + student 同时驻留）；本地只用 `--selfcheck` 验逻辑。

---

## 3. 推荐执行顺序（Q2 节点 OPD，理由见上一轮分析）

```
0.8B: Q1 SFT → Q2 SFT ─────────────────────────────┐
4B  : Q1 SFT → Q2 SFT → GRPO_Q2(定位优先) ──────────┤
                                                     ├─ OPD(teacher=4B-GRPO-Q2) → 0.8B
                                                     └─ 评测（同 val/batch/分辨率）
```

1. **先量 gap**：4B-GRPO-Q2 vs 0.8B-Q2 的 `source_hit` / `xml_well_formed`。
   - > 8pp → OPD 值；3–8pp → 考虑「Q1 OPD → 再跑 Q2 SFT」；< 3pp → 别做。
2. **数据**：就用 `data/splits_grpo_q2_hq200/`（同分布）或全量 Q2 train 中筛出的硬样本，做学生 rollout。
3. **训练**：`VLGKDTrainer`，`lmbda=1.0`、`beta=1.0`、`temperature=1.0`、`max_new_tokens=256~512`、`lr≈1e-6~2e-6`（比 SFT 小一个量级）、LoRA 与 SFT 同构。
4. **loss mask**：只蒸馏 `<source>` 与 XML 标签段（把 `<translation>` 段的权重降到 ~0）——理由：teacher 的翻译也只强一点点，
   学生容量不该花在那里。实现：在 collator 里把 translation 段的 `labels` 置 -100（TRL 的 loss 只在 labels≠-100 处算，见 §2.2）。
5. **门禁**：`source_hit` ≥ +2pp、`xml_well_formed` 不降、`effective_empty` 不降；翻译 chrF 提升 <0.02 不算 OPD 的功劳。

---

## 4. 显存与工程要点（32GB）

- 顺序执行即可：**学生 rollout（generate）→ 存轨迹 → teacher 前向打分 → 学生反传**。TRL 的 `training_step` 内部就是
  「先 generate 再 forward」，同一 step 里 teacher 前向用 `no_grad`，所以峰值 ≈（学生 + teacher 权重）+ 一次 completion 段 logits。
- 4B-4bit ≈ 2.3GB + 0.8B-4bit ≈ 0.5GB + 8k 图 token 的 KV/激活 → 32GB 够；**8GB 不行**（teacher+student 同时驻留）。
- 若不想改 TRL：**备选 C** 的自研循环里，student/teacher 可以分进程分阶段跑（rollout 落盘 → teacher 打分落盘 → 学生训练），
  代价是工程复杂度，收益是能精确实现论文的 top-k 稀疏化。

---

## 5. 立即可做的一件小事

如果之后要做**两阶段（纯文本翻译）**，TRL `GKDTrainer` **不需要任何修改**（无图）：
```python
GKDConfig(lmbda=1.0, beta=1.0, temperature=1.0, max_new_tokens=256,
          teacher_model_name_or_path="<4B-Q2-merged>", ...)
```
学生 = 0.8B（同一权重），数据 = `(source → translation)` 文本对（用 `scripts/relocate_paths.py` 同款的
「无图样本」能力，代码已支持 `image=None`）。

---

## 6. 待办

- [ ] 实现 `train/unsloth_gkd_vl.py`（`VLGKDTrainer` 子类 + VL collator + `prompts` 构造）并加单测
- [ ] translation 段 label mask（只蒸馏 source/XML 段）
- [ ] 先跑 **gap 测量**，再决定是否启动 OPD

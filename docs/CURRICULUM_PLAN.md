# Stage A 课程计划（新一轮 · A1 → A2）

从基模重训两轮 POINT。吸收旧 A1/A2 复盘：bf16 全精度、早停、固定品红 45° X 准星、砍掉桌面/贴边硬指标；死磕长块、短块边界、块角、空输出。

> **状态（2026-09-16）** 本文件是上一轮 A1→A2 配方，**不再作为执行清单**。  
> 下一轮见 [`ROUND_B_PLAN.md`](ROUND_B_PLAN.md)。A1 权重 `20260915_205241_curriculum_a1` 保留为续训 / GRPO 底座；A2 / A2-b 不毕业。

---

## 0. 总原则

1. **A1 打地基**（跟点、单长块、该空就空、压 PAGE）；**A2 补难例**（邻接、multi-frag、短块整组边界、块角），并 **回放 A1** 防回退。  
2. **同一阶段 train ≡ eval ≡ 产品** 共用一套准星算法与 `POINT_PROMPT`。  
3. **造数 → 分层预览 → 你验收 → 再全量 → 再训**。  
4. 桌面壳 / 整页贴边：**不作为毕业硬指标**（产品可规避）；造数占比 ≤ 2% 或不造。  
5. 基座加载：**原始精度（bf16），禁止默认 4bit QLoRA**。  
6. **早停必须开**；跑满 epoch 不是目标。

---

## 1. 旧轮复盘（极简）

| | 旧 A1 | 旧 A2 |
|--|-------|-------|
| 设定 | 基模 · vision 开 · **r32** · QLoRA · 十字准星 | 续训 · 冻 vision · lr 偏低 · 散点星 · 难例堆太杂 |
| 结果 | 长块/空输出较好 | desktop≈0 · semantic over≈58% · PAGE 回潮 |
| 主因 | 任务可做，但 4bit 扭矩有限；邻接/短块未训满 | 数据同质 + 扩上下文串味 + 冻视觉/换准星 |

**你指定的 A1 训练超参（本轮采用）：lr = `5e-5`，LoRA r = `32`（α = `64`）。**  
（磁盘上旧 run `20260914_000558` 的 `run_config` 记的是 `8e-5`；以本配方为准开新跑。）

---

## 2. 准星协议（A1 / A2 共用）

| 项 | 规格 |
|----|------|
| 形状 | **品红十字旋转 45°（X）** |
| 透明度 | **不透明**（α=255；白描边/白环同样不透明） |
| 尺寸 | 固定像素：臂半长 34px、线宽 3px、白环 r=28 / 宽 3px；**不随图像缩放** |
| 颜色 | **固定品红 `#FF00FF`**，不随底色自适应 |
| 实现 | `MARKER_SPEC` + `draw_crosshair()`；两阶段同一套 |

Prompt 写死颜色与形状：*magenta-and-white cross rotated 45 degrees (an X-shaped interest point)*。

---

## 3. 不可妥协能力（两阶段共同门禁）

| ID | 能力 | 主要落在 |
|----|------|----------|
| **L** | 简单长语义块（连续单 rect） | A1 主训 · A2 回放 |
| **M** | 多 fragment → 整段 | A2 主训 · A1 可少量预热 |
| **S** | 短块上下文：不缺相关、不多无关 | A2 主训 |
| **C** | 准星在块角仍出对块 | A2 主训 · A1 可少量 |
| **E** | 非语义区 → 空 | A1 打底 · A2 保持 |

不做硬指标：桌面多窗、任务栏、壁纸、极端贴边 chrome。

---

## 4. A1 — 地基（从基模）

### 4.1 目标

1. 扭 PAGE → POINT：跟圆点出 **恰好一块**，禁止整页 /「第一块惰性」。  
2. 长段落（center_band）高 hit。  
3. 空白 / 栏缝 / 页眉页脚装饰 → **空**。  
4. 为 A2 留接口：允许 **少量** multi-frag 与块角，但不作为 A1 毕业主指标。

### 4.2 数据（建议 **15k**）

路径：`data/processed/a1_v2/` → `data/splits_a1_v2/`（90/5/5，打散）。

| 切片 `a1_slice` | 占比 | ≈@15k | 说明 |
|-----------------|------|--------|------|
| **long_center** | 62% | 9300 | 大块段落；center_band ±0.35；面积/字数门槛 |
| **adjacency_light** | 10% | 1500 | 轻量邻接，降低 A2 跳跃 |
| **multi_frag_light** | 6% | 900 | 少量多 fragment 预热 |
| **corner_light** | 4% | 600 | 少量块角预热 |
| **empty_neg** | 18% | 2700 | 空白/栏缝/装饰 → 空（**无桌面壳**） |
| **合计** | 100% | 15000 | 负例≈18% |

模板：文档向（`01/04/08/12/14/20/24` 等）；**禁止**桌面壳进 A1 主配额。  
每页多点时 **强制中/下/右栏配额**，避免总采阅读顺序第一块。

### 4.3 训练

| 项 | A1 |
|----|----|
| 起点 | **基模** `ATH-MaaS/OvisOCR2`（无 adapter） |
| 精度 | **bf16，`load_in_4bit=False`** |
| Vision LoRA | **开** |
| rank / α | **32 / 64** |
| lr | **`5e-5`** |
| epochs | max **2** + **早停 patience=4**（监控 eval_loss） |
| batch | 显存优先：`1 × accum 8` 或 `2 × accum 4`（有效≈8） |
| max_pixels | `2880²`（OOM 则先 `2048²`） |
| 其它 | `load_best_model_at_end=True`；cosine + warmup≈3% |

```bash
uv run python train/unsloth_stage_a.py \
  --data data/splits_a1_v2/train.jsonl \
  --val data/splits_a1_v2/val.jsonl \
  --model ATH-MaaS/OvisOCR2 \
  --no-4bit \
  --finetune-vision \
  --lora-rank 32 --lora-alpha 64 \
  --lr 5e-5 --epochs 2 \
  --early-stopping-patience 4 \
  --batch-size 2 --grad-accum 4 \
  --max-pixels $((2880*2880)) \
  --min-pixels $((448*448)) \
  --save-steps 200 --eval-steps 200
```

### 4.4 A1 毕业

| 指标 | 目标 |
|------|------|
| overall hit（正例） | ≳ **85%** |
| over_extraction | ≲ **3%** |
| empty_on_neg | ≳ **92%** |
| format_leak | **0** |
| 目视 | 少见整页倾倒 / 总出第一块 |

未毕业：先查负例比例与第一块采样偏差，再调 lr∈[3e-5, 8e-5]，不进入 A2。

---

## 5. A2 — 难例补强（续训 A1）

### 5.1 目标

在 **不毁掉 A1 长块/空输出** 的前提下补齐：

1. 邻接不串台  
2. 多 fragment 整段  
3. **短块整组**：双向追溯；**禁止组外续写、禁止复读、禁止把 packing 用到长块**  
4. 块角仍跟块  
5. 非语义区继续空（可含少量公式/代码拒识）

**明确不做：** 桌面壳、整页贴边毕业。

### 5.2 数据（建议 **16k**）

路径：`data/processed/a2_v2/` → `data/splits_a2_v2/`。  
**全部样本用 A1 同一套准星重打标**（含回放）。

| 切片 `a2_slice` | 占比 | ≈@16k | 说明 |
|-----------------|------|--------|------|
| **replay_long** | 26% | 4160 | A1 风格长块回放（防回退） |
| **adjacency** | 14% | 2240 | 双栏/邻段；edge_band |
| **multi_frag** | 14% | 2240 | 点任一 frag → 整段 |
| **semantic_group** | 20% | 3200 | 短块整组；见 §5.3 |
| **corner_extreme** | 10% | 1600 | 块角 |
| **empty_neg** | 12% | 1920 | 空白/栏缝/装饰 → 空 |
| **special_light** | 4% | 640 | 表→整表 HTML；公式/代码/图→空 |
| **合计** | 100% | 16000 | 负例约 12–16% |

划分：90/5/5；`--stratify-key a2_slice --interleave-template`。

### 5.3 短块（semantic_group）硬规则

- **≥5 个版式模板** + 文案池；禁止只靠单一 `27`。  
- seed≈40% / body≈60%；GT = 整组 Markdown。  
- **截断负例（必做）**：点在组外紧邻 → GT=该外块或空（全局固定一种）；专治过提取。  
- 长块切片 **不得** 使用整组扩写标签。  
- Val 单独报：hit、over、缺 seed 率、组外续写率。

### 5.4 训练

| 项 | A2 |
|----|----|
| 起点 | **A1 `adapter_final` 续训** |
| 精度 | **bf16，无 4bit** |
| Vision LoRA | **开**（换难例布局仍要视觉；勿再冻 vision） |
| rank / α | **保持 A1 的 32/64**（直接加载，勿改 rank） |
| lr | **`3e-5` ~ `5e-5`**（续训低于 A1；默认 **4e-5**） |
| epochs | max **2** + **早停 patience=3~4** |
| batch | 同 A1 量级 |
| 监控 | replay_long hit 回退 ≲ 5～8 pp 则停/降 lr |

```bash
uv run python train/unsloth_stage_a.py \
  --data data/splits_a2_v2/train.jsonl \
  --val data/splits_a2_v2/val.jsonl \
  --model ATH-MaaS/OvisOCR2 \
  --adapter checkpoints/<A1_RUN>/adapter_final \
  --no-4bit \
  --finetune-vision \
  --lora-rank 32 --lora-alpha 64 \
  --lr 4e-5 --epochs 2 \
  --early-stopping-patience 4 \
  --batch-size 2 --grad-accum 4 \
  --max-pixels $((2880*2880)) \
  --min-pixels $((448*448)) \
  --save-steps 200 --eval-steps 200
```

### 5.5 A2 毕业

| 子集 | 目标 |
|------|------|
| replay_long hit | ≳ 80%（相对 A1 回退 ≲ 8 pp） |
| multi_frag hit | ≳ 70% |
| semantic_group hit | ≳ 70%；**over ≲ 8%** |
| corner hit | 与 center 差 ≲ 8 pp |
| empty_neg empty | ≳ 90% |
| overall over | ≲ 4% |
| format_leak | 0 |
| 目视短块 | 无系统性缺上下文 / 组外续写 / 长块被 packing |

---

## 6. Prompt

### A1

```
The image contains a prominent magenta-and-white cross rotated 45 degrees
(an X-shaped interest point).
Identify the single minimal semantic block under that mark
(e.g. one paragraph, heading, or list item).
Output ONLY that block as Markdown.
If the mark is on blank space, chrome, decoration, or non-text, output an empty string.
Do NOT dump the full page, other columns, or neighboring blocks.
```

### A2（在 A1 上只加边界规则）

```
The image contains a prominent magenta-and-white cross rotated 45 degrees
(an X-shaped interest point).
Identify the single semantic unit under that mark:
- a normal text block → that block only;
- a short anchor inside a tight semantic group → the whole group only;
- blank / chrome / decoration / formula / code / image → empty string.
Output ONLY that unit as Markdown (tables as HTML <table>...</table>).
Do NOT dump the full page, other columns, or text outside the unit.
```

---

## 7. 工程清单（开训前）

| 项 | 说明 |
|----|------|
| 准星 | 固定品红 X（45°）+ demo 验收 |
| `train/unsloth_stage_a.py` | `--no-4bit`、`--early-stopping-patience`、`--adapter`（已有则复用） |
| `build_a1_dataset_v2.py` / `build_a2_dataset_v2.py` | 平行目录；切片配额；打散 |
| 语义组模板 | ≥5 版式 + 截断负例 |
| 预览 | `review_sample_gallery.py --stratify a*_slice` |
| 早停 | patience + best ckpt；日志旁路分层 hit/over（可后置脚本） |

---

## 8. 执行顺序

1. 准星 x45 demo → 你验收  
2. 训练脚本 bf16 + 早停  
3. 造 A1 预览 → 全量 15k → `splits_a1_v2`  
4. **A1 基模训**（lr `5e-5`，r32，vision 开）→ 毕业  
5. 造 A2 预览（含回放+语义组多模板）→ 全量 16k → `splits_a2_v2`  
6. **A2 续训**（adapter=A1，lr `4e-5`，vision 仍开）→ 毕业门禁  
7. 产品切换到 A2 ckpt；旧 A1/A2 目录仅归档

---

## 9. 超参速查

| | A1 | A2 |
|--|----|----|
| 起点 | 基模 | A1 adapter |
| 精度 | bf16 | bf16 |
| vision LoRA | 开 | 开 |
| r / α | **32 / 64** | **32 / 64**（继承） |
| lr | **`5e-5`** | **`4e-5`**（可 3e-5～5e-5） |
| max epochs | 2 | 2 |
| 早停 patience | 4 | 3～4 |
| 规模 | 15k | 16k |
| 准星 | 固定品红 45° X | 同左 |

---

## 10. 一句话

**A1 用 bf16 + r32 + lr 5e-5 从基模把 POINT 扭稳（长块+空）；A2 用同一准星续训、仍开 vision、略降 lr，专补邻接/multi-frag/短块边界/块角，并用回放与早停防止低 loss 过拟合与 PAGE 回潮——桌面和贴边先不做。**

# Round B — 壳层鲁棒 + 五区采样 + 不透明 X + A1 续训 / GRPO

日期：2026-09-16。底座：**A1** `checkpoints/20260915_205241_curriculum_a1`。不从 A2 / A2-b 续训。

Studio 探伤入口：`uv run python eval/marker_studio/server.py` → http://127.0.0.1:7865

---

## 0. Studio 结论（锁定）

1. **窗口壳是主 OOD。** 带浏览器 / WPS / Word 顶栏的真实截图明显差；裁掉壳、同一点、同一解码立刻变好。当前 synth 几乎是「客户区正文」，模型把壳上第一行可读文字当成 PAGE 先验。
2. **块内位置敏感。** 现 `core_inner` 每块只抽 1–2 个点，且在 inner-80% 矩形里均匀撒，角上几乎没见过。要对齐产品「点在块内各处都出这一块」。
3. **半透明 X 模型认不出。** 体积反比 α 方案放弃。协议改为 **α=255**；挡字换准确。
4. **GRPO 有抓手。** `T=1.0, top_p=1.0, n=8` 多数能同时抽出正确块、邻块、空串。先把壳和角送进 SFT，再 RL，避免把「读壳」reinforce 进去。

产品仍不裁切：训练图必须带壳，推理也带壳。

---

## 1. 准星协议 `x45c`

| 项 | 值 |
|----|-----|
| 形状 | 品红 45° X + 白描边 + 白环（不变） |
| 面积比 | **0.5%**（对角线正方形 / 整图），参考 2582×1641 → 臂长 ≈206px |
| 不透明度 | **恒 α=255**（十字 / 描边 / 环 / 心点）。删掉体积反比 α |
| 实现 | `spec_for_image` + `draw_crosshair`；studio 默认 α=auto 也走 255 |

改协议后 **必须重打标**，不能只改 jsonl 字段。

---

## 2. 五区采样（正样本）

矩形 bbox 四边中点连成菱形，把矩形分成 **五块：NW / NE / SW / SE 角 + 中心菱形**。

菱形判定（中心 `(cx,cy)`，宽 `w`、高 `h`）：

```
|x-cx|/(w/2) + |y-cy|/(h/2) ≤ 1  →  diamond
其余按象限归四个角
```

**切分作用在 inner-80% 矩形上**，不是 ink box 外沿。菱形顶点落在 inner 边中点，全部正点仍满足产品「inner 80% → 该块」；若切全框，边中点会落到 inner 外，GT 应对空。

每个被标注的文本块：

- 从五区中 **至少抽 3 区**（无放回；优先保证 diamond 常在，再随机角）。
- 每区 **均匀随机 2 点**（区内 rejection 或三角形映射）。
- `region` 写入 meta：`diamond` / `nw` / `ne` / `sw` / `se`。
- 多 fragment：对 **被点中的那一块 fragment** 做五区，GT 仍是整段。
- `semantic_group`：对 **union bbox** 做五区（含组内 gutter 的菱形中部）。

负样本不变：`empty_clear` / `empty_boundary`，另加壳上的空标（见 §3）。

当前缺口：`emit.py` 里 `core_inner` 是 `r_min=1, r_max=2, coverage="inner_area"`。改为 `coverage="diamond5"`，每块 6 点（3×2）。实现落在 `sample_points.py`，单测覆盖：五点分区互斥且并集=inner 矩形、每区 2 点、至少 3 区。

---

## 3. 窗口壳（模板）

不要为每个文档再复制一套 HTML。渲染时 **套一层壳**（Playwright 仍拍整个 viewport）。

| 家族 | 壳头 | 壳尾 | 壳边 | 空标点 |
|------|------|------|------|--------|
| `browser` | 标签 + 地址栏 + 书签条 | 下载/状态条（可选） | 滚动条 | 点在按钮/URL/标签 → `""` |
| `wps_word` | 标题栏 + 功能区 | 状态栏（页码/字数） | 标尺 / 导航窄条 | 点在功能区 → `""` |
| `none` | 无 | 无 | 无 | 对照「裁壳就好」的客户区 |

- 壳体是 **第三层**，与内容多样性、难例池正交：同一套模板/池既可以 `none` 也可以套壳。
- 预览与日后全量都按权重抽样，默认约 **45% 无壳 / 35% 浏览器 / 20% 办公**，禁止「池子里全是带壳页」。
- 壳上的字（「搜索」「开始」「布局」）**不是**语义块：不要 `data-block-id`。
- 正文 `data-block-id` 只在客户区。
- `empty_clear` 在带壳页上额外抽壳带空标（头/尾/边）。
- `core_inner` 正点仍在正文上，但带壳页的像素里看得到壳。

桌面多窗 / 壁纸仍不做毕业硬指标。

---

## 4. 数据怎么造

五区改的是 **点**，壳改的是 **像素**。只 `remake_pool_markers.py` 不够，要 **重渲染 + 重 emit**。

建议规模（保持页多样性，接受正点变密）：

| 池 | 作用 | 规模（约） |
|----|------|------------|
| `core_inner` | 带壳正文 + 五区 | 18k |
| `empty_clear` | 空白 + **壳带空标** | 8k |
| `empty_boundary` | 邻接歧义空 | 4k |
| `multi_frag` | fragment 五区 | 4k |
| `semantic_group` | union 五区 | 4k |
| **compose A1-B** | 接近原 A1 配比，略提高 empty_clear | **train 22k / val 1.2k / test 1.2k** |

路径：`data/pools_b/`、`data/splits_stage_b/`，**不覆盖**现有 `data/pools/` 与 A1 jsonl。

流程：壳包装 → Playwright → 五区 emit → compose → 分层预览（含带壳 / 五区散点图）→ 你过目 → 全量。

---

## 5. 训练顺序（不要再走 A2 配比）

```
A1 adapter_final
    → SFT-B（新池，含壳 + 五区 + 不透明 X）
    → studio 门禁（真实截图，带壳）
    → 通过后再 GRPO（仍从 SFT-B，不从 A2-b）
```

**SFT-B**

- 续训 A1 LoRA（r32/α64，vision LoRA 开，bf16）。
- lr `3e-5`–`5e-5`，1–2 epoch，batch 4×accum 4，早停看 val hit + 空串，防 PAGE 回潮。
- 不把 A2 难例配比拉满；壳和角是这一轮主课。

**Studio 门禁（过了再 RL）**

同一张真实图、同一点、greedy：

- 带壳 vs 裁壳：带壳不再崩成地址栏 / 功能区。
- 同一块的 diamond / 一角 / 对角：都应出该块，而不是换块。
- α=255 协议尺寸；采样 `T=1, top_p=1, n=8` 仍能抽出正确轨迹。

**GRPO（门禁后再开）**

- 解码：`temperature=1.0`, `top_p=1.0`, top_k 关, 重复惩罚 1.0。
- G：标准 8；8GB 先 G=2–4。
- Reward 草案：block hit / 编辑相似；空标要对空；壳 dump 重罚；超抽（整页）重罚。
- 8GB：4bit + freeze vision + 小 completion（256）先做通，再谈放大。

---

## 6. 执行清单

1. README / studio 已接入（本轮文档）。
2. `marker.py`：α 恒 255，tag `x45c`；改测试与 `MARKER_SPEC.md`。
3. `sample_points.py`：`coverage="diamond5"` + 单测。
4. `emit.py` / `build_point.py`：正样本走五区；meta 记 region。
5. 壳包装：`browser` / `wps_word` / `none`，渲染注入；壳带抽空标。
6. 预览 ~50 页（带壳 + 五区可视化）→ 你验收。
7. 全量 `pools_b` + `splits_stage_b`。
8. SFT-B 从 A1 续训。
9. Studio 真实截图门禁。
10. 通过后 GRPO。

未完成 9 之前不开 RL。不覆盖 A1 checkpoint。

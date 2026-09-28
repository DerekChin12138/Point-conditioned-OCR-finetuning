# 样本池管线（目标池造数 + 阶段配比抽样）

两层分离：

1. **模板多样性**：窗口 3:4–16:9；页面里可以出现表 / 公式 / 代码，作为画面背景。  
2. **目标池**：只按「点落在哪、GT 是什么」造样本。标记**不打在**表 / 公式 / 代码上。

阶段只改 YAML 配比，不重渲图。

---

## 判定规则

| 情形 | 点在哪 | GT |
|------|--------|----|
| **命中** | 文本 bbox **内部 80% 面积**（边内缩 ≈5.3%） | 该语义单元 Markdown |
| **空 · 明确** | 远离所有墨迹（含 special 墨迹） | `""` |
| **空 · 交界** | 两**单元**缝/贴边，且不在任一内部 80%；也不落在 special 墨迹上。相关短块先合成一个 bbox，组内缝不算交界 | `""` |
| **换行/换列** | 落在该段任一 fragment 内部 80% | **整段** |
| **短块组** | 落在组并集 bbox 内部 80%（含成员之间的缝） | **整组**；组外邻居只出自己 |
| **块角 / 外圈 20%** | 不算所属 | 消化为空（无正例池） |
| **PAGE 整页倾倒** | — | 不进池；留给最后小幅 RL |

---

## 五个池（库存）

| `pool_id` | 规模 | 说明 |
|-----------|------|------|
| `core_inner` | **15k** | 内部命中 → 单块；模板含文档 + special 背景 |
| `empty_clear` | **7.5k** | 明确空 |
| `empty_boundary` | **5k** | 交界空 |
| `multi_frag` | **5k** | fragment → 整段 |
| `semantic_group` | **5k** | 相关短块整组 |

磁盘：`data/pools/<pool_id>/`

---

## 阶段配比

**A1（15k）** — 打地基，难例只是先见过：

| 池 | 占比 | ≈条数 |
|----|------|------|
| core_inner | 60% | 9000 |
| empty_clear | 25% | 3750 |
| empty_boundary | 5% | 750 |
| multi_frag | 5% | 750 |
| semantic_group | 5% | 750 |

**A2（16k）** — 开始抠难例正确率：

| 池 | 占比 | ≈条数 |
|----|------|------|
| core_inner | 40% | 6400 |
| empty_clear | 15% | 2400 |
| empty_boundary | 15% | 2400 |
| multi_frag | 15% | 2400 |
| semantic_group | 15% | 2400 |

A1 / A2 均用 `prompt_key: a2_v2`（单块或紧密短组）。compose 按配方写入 user 文本。

---

## 命令

```bash
uv run python data/scripts/build_pool.py --all --wipe --workers 30
uv run python data/scripts/compose_stage.py --recipe data/recipes/stage_a1.yaml
# → data/splits_stage_a1/{train,val,test}.jsonl
```

# 训练框架说明（详解）

面向「把本仓库搬到云端 GPU 自行跑通」的读者。配套 notebooks 见 `notebooks/`。

## 0. 一句话目标

在 **OvisOCR2**（整页 → 整页 Markdown）之上，微调出 **POINT 能力**：图上有固定品红/白准星时，模型只输出准星所在的**最小语义块**；准星在空白/UI chrome 上则输出空字符串。

零样本改提示几乎无效（仍会倾倒整页），所以必须专门造数据 + SFT。

---

## 1. 仓库分层（数据流）

```
HTML 模板 / 真实截图
        │
        ▼
┌───────────────────┐
│ 块注解 (bbox+文)   │  合成：DOM bbox + HTML→Markdown
│                   │  真实：布局模型 bbox + OvisOCR2 裁剪 OCR
└─────────┬─────────┘
          │ 每块采 R=2~5 个点 + 若干负例
          ▼
┌───────────────────┐
│ 原图叠 MARKER_SPEC │  → 带准星的 JPEG
│ + 固定 POINT 提示  │  → ShareGPT JSONL
└─────────┬─────────┘
          │ merge → LLaMA-Factory dataset
          ▼
┌───────────────────┐
│ Stage A QLoRA SFT │  冻 vision，学「只吐一块」
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ held-out 三项指标  │  hit ↑ / over-extraction ↓ / empty-on-chrome ↑
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ merge LoRA → GGUF │  (+ mmproj 若部署需要)
└───────────────────┘
```

| 目录 | 职责 |
|------|------|
| `src/point_ocr/` | 可 import 的核心库（准星、提示词、采样、指标、造样本） |
| `data/synth/templates/` | 带 `data-block-id` 的 HTML 难例模板 |
| `data/scripts/` | CLI 批处理（notebook 会调用同样逻辑） |
| `train/` | LLaMA-Factory YAML |
| `eval/` | held-out 构建与打分 |
| `export/` | LoRA 合并与 GGUF |
| `notebooks/` | **云端逐步执行入口（推荐）** |

`.py` 仍保留：便于 CI/批跑；你在云端优先点 `notebooks/`。

---

## 2. 核心协议（训练 ≡ 推理，必须一致）

### 2.1 准星 `MARKER_SPEC`（`marker.py`）

- 品红十字 `#FF00FF` + 白描边 + 白色圆环 + 中心点  
- 几何偏大，避免 resize/压缩后「准星消失」导致模型学不到条件  
- **唯一画法**：`draw_crosshair(image, x, y)`  
- 产品侧以后也必须用同一套参数画准星，否则分布偏移

### 2.2 提示词（`prompts.py`）

- **POINT**：强调「全图有准星 → 只输出含准星的最小块 → 禁止整页 → 无字则空」  
- **PAGE**：官方 OvisOCR2 整页 Markdown 指令（仅 Stage B）  
- 样本里 user 文本必须来自这里，不要手写变体（除非刻意增广且评估过）

### 2.3 单条训练样本长什么样

ShareGPT 多模态格式（LLaMA-Factory）：

```json
{
  "messages": [
    {"role": "user", "content": "<image>……POINT_PROMPT全文……"},
    {"role": "assistant", "content": "仅该块的 Markdown，或空字符串"}
  ],
  "images": ["/abs/or/rel/path/to/marked.jpg"]
}
```

输入永远是**整页 + 准星**，不是裁剪块；输出才是单块。这样模型学会「在整页里定位」。

---

## 3. 数据引擎细节

### 3.1 为什么「一图多样本」是硬性的

同一页有几十个块 → 每块采 2~5 个点（中心偏置 + 抖动）→ 再加页边/栏缝负例。  
一页可膨胀到近百条，且强迫模型依赖准星位置，而不是背「这一页的全文」。

实现：`sample_points.py` + `build_point.py`。

### 3.2 合成臂（优先放量）

1. 写 HTML，块上打稳定 `data-block-id`  
2. Playwright 渲染整页截图  
3. JS `getBoundingClientRect` 取 bbox（× deviceScaleFactor）  
4. **标签 = 同源 HTML 序列化 Markdown**（`html_to_md.py`）——禁止对合成图再跑 OCR 当 GT  
5. 过滤坏标签（`filter_qa.py`）→ 叠准星 → JSONL  
6. 可选屏幕噪声：JPEG / 缩放（`noise.py`）

模板起步：`data/synth/templates/*.html`（双栏、表格公式、代码 UI）。

### 3.3 真实臂（补分布，约 15–30%）

1. 脱敏桌面/浏览器截图、自渲染 PDF 页等  
2. 布局模型出 bbox（需你接入真实 detector；现有 `StubLayoutDetector` 仅烟雾测试）  
3. **OvisOCR2 对每块 crop 造文**（`real/ovis_crop_client.py`）作标签首选  
4. 原图多点画准星成对  

版权与隐私：训练自用，不要随便公开原图。

### 3.4 合并

`merge_and_filter.py`：合成 + 真实按比例混合 → `data/llamafactory/ovisocr2_point_{train,val}.json` + `dataset_info.json`。

Stage B 另用 `build_dual_task.py` 混 PAGE（无准星整页图 + 全文）与 POINT。

---

## 4. 训练（Stage A）

### 4.1 为什么是 LoRA / QLoRA，不是论文全流程

论文有 SFT → 4B GRPO → OPD 回 0.8B → 融合。  
单卡 ~24GB 默认：**只做 POINT 向 SFT**，冻 vision tower，让语言模型学会约束输出。

配置：`train/stage_a_point_qlora.yaml`  
启动依赖外部 **LLaMA-Factory**（本仓不 vendoring 整套训练框架，避免重复造轮子）。

注意：YAML 里 `template: qwen3_vl` 需在云端对照 OvisOCR2 的 `config.json` / LLaMA-Factory 支持列表核对；不对就改成该模型实际 template 名。

### 4.2 建议超参直觉

- `per_device_train_batch_size=1` + `gradient_accumulation_steps=16`  
- `lora_rank=16`，`lr=1e-4`，`epochs=1~3`  
- **早停看 over-extraction**，不是只看 train loss（loss 降了仍可能整页倾倒）

### 4.3 Stage B

仅 Stage A 达标后再试双任务。互相干扰 → 放弃单权重，保留 POINT 独立权重。

---

## 5. 评测

先建 held-out，再大规模训练。

| 指标 | 含义 |
|------|------|
| `block_hit_rate` | 正例预测与 GT 编辑相似度 ≥ 阈值 |
| `over_extraction_rate` | 像整页倾倒 / 远长于 GT |
| `empty_on_chrome_rate` | 负例应输出空 |

对比基线：原版 `ATH-MaaS/OvisOCR2` + 同一 POINT 提示、同一准星图（应明显更差）。

---

## 6. 导出

1. `export/merge_lora.py`：adapter merge 进基座  
2. `export/export_gguf.sh`：llama.cpp 转 GGUF（Q4_K_M / Q5_K_M）  
3. 若推理栈要单独 mmproj，按同版 llama.cpp 再导视觉投影

---

## 7. 迁到云端租卡（实操清单）

### 7.1 机器选型

- GPU：24GB 级（如 4090 / A10 / L4 / A5000）；多卡非必须  
- 系统：Ubuntu 22.04 + NVIDIA 驱动 + CUDA（与 torch/vLLM 版本匹配）  
- 磁盘：模型 + 数据建议 ≥ 100GB（合成图会很大）

### 7.2 把本仓库弄上去

任选：

```bash
# A. git（推荐）
git clone <你的远程> && cd point-conditioned-ocr-finetuning

# B. 本地打包上传（无 git 时）
# 本地：排除 .venv
tar --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  -czf point-ocr.tgz point-conditioned-ocr-finetuning
# 云端：解压后继续
```

**不要**上传本机 macOS 的 `.venv`。云端用 uv 重建。

### 7.3 云端环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd point-conditioned-ocr-finetuning
uv sync --extra dev --extra synth
uv run playwright install chromium
# 训练日再：
uv sync --extra train
# 评测/造真实标签再：
uv sync --extra eval   # 需 CUDA 且注意 vLLM 与驱动匹配
```

另装 **LLaMA-Factory**（独立 clone），设：

```bash
export LLAMA_FACTORY_ROOT=/path/to/LLaMA-Factory
```

### 7.4 Jupyter

```bash
uv pip install jupyter ipykernel
uv run python -m ipykernel install --user --name=point-ocr --display-name="point-ocr"
uv run jupyter lab --ip=0.0.0.0 --port=8888
# 用 SSH 隧道或云厂商提供的 Notebook 入口打开
```

按顺序打开：

1. `notebooks/00_cloud_setup.ipynb`  
2. `notebooks/01_marker_demo.ipynb`  
3. `notebooks/02_build_synth_data.ipynb`  
4. `notebooks/03_merge_for_llamafactory.ipynb`  
5. `notebooks/04_train_stage_a.ipynb`  
6. `notebooks/05_eval.ipynb`  
7. `notebooks/06_export.ipynb`  

### 7.5 Hugging Face 权重

云端需能拉 `ATH-MaaS/OvisOCR2`（或先在可联网机器 download 再 rsync）。  
如需 token：`huggingface-cli login`。

### 7.6 常见坑

| 现象 | 排查 |
|------|------|
| 仍整页倾倒 | 准星是否够大；负例是否太少；标签是否整页误标 |
| OOM | 开 gradient checkpoint；确认 freeze vision；试 quantization_bit=4 |
| template 报错 | 改 YAML `template` 对齐 OvisOCR2 |
| Playwright 无显示 | 用 headless chromium；云镜像装依赖 `playwright install-deps` |
| 路径不对 | ShareGPT 里 `images` 用云端绝对路径或相对 `dataset_dir` 可解析路径 |

---

## 8. 建议的执行节奏

1. 模板 3 页冒烟 → 目视带准星图与标签  
2. 手检扩到 ~1k 对 → 建 held-out → 记零样本基线  
3. Stage A 短训 → 看 over-extraction 是否下降  
4. 数据放量 5万–20万 → 再训 → 导出 GGUF  
5. （可选）Stage B  

验收：同屏同准星，输出应「主要是那一块」；否则先修数据/标记，不盲目换更大基座。

# AutoDL × Unsloth × OvisOCR2（Qwen3.5）微调配置备忘

> 记录日期：2026-09-11  
> 场景：点条件语义块 OCR（POINT）Stage A 首次全量试跑  
> 目的：把「能跑通」的环境、依赖、造数、训练命令和踩坑固化成可复用配方，供后续同类 VL 微调直接套用。

仓库：`DerekChin12138/Point-conditioned-OCR-finetuning`  
基座：`ATH-MaaS/OvisOCR2`（`model_type: qwen3_5`，`Qwen3_5ForConditionalGeneration`）

---

## 1. 总览：这套栈是什么

| 层级 | 选型 |
|------|------|
| 云主机 | AutoDL，Linux + NVIDIA（约 24GB 级） |
| 包管理 | **uv**（有 pip 也可，但本配方以 uv 为准） |
| 训练框架 | **Unsloth** `FastVisionModel` + TRL `SFTTrainer` |
| 基座 | OvisOCR2 / Qwen3.5-VL 系（需 **transformers ≥ 5.2**） |
| 数据 | 合成 HTML → Playwright → 准星监督对（**不依赖 LLM 造标**） |
| 首次全量超参 | **保守**：1 epoch、较小 LoRA、较短 seq |

Notebook（`notebooks/PointOCR_Full_Pipeline.ipynb`）可选；终端流程已足够。

---

## 2. 弱网镜像（每个新终端先 source）

```bash
cd /root/autodl-tmp/ocr-finetune/Point-conditioned-OCR-finetuning  # 按实际路径改
source scripts/setup_autodl_mirrors.sh
export HF_HUB_DISABLE_XET=1          # 避免 HF Xet 401；强烈建议常开
export HF_ENDPOINT=https://hf-mirror.com
```

脚本默认覆盖：

- PyPI / uv：阿里云 + 清华  
- Hugging Face：`hf-mirror.com`  
- Playwright：npmmirror  
- GitHub clone 提示：`ghproxy.net` + `git -c http.version=HTTP/1.1`

> **注意：** 哪怕模型参数已下载到本地缓存，开训前仍要保证 Hugging Face（镜像）连通。`from_pretrained` 仍可能访问 Hub；断网时常见卡住或奇怪报错。

Git clone 示例（HTTP/2 framing 报错时）：

```bash
git -c http.version=HTTP/1.1 clone --depth 1 \
  https://ghproxy.net/https://github.com/DerekChin12138/Point-conditioned-OCR-finetuning.git
```

---

## 3. 环境安装（uv）

```bash
# 安装 uv（若尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

source scripts/setup_autodl_mirrors.sh
export HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com

uv sync --extra dev --extra synth
uv run playwright install chromium
uv run playwright install-deps chromium   # Linux 系统库；AutoDL/本机 Ubuntu·WSL 缺 libgbm 时必做（跑完即可正常造数）

uv sync --extra train
uv pip install unsloth unsloth_zoo
```

### 3.1 关键依赖版本（2026-09 更新）

OvisOCR2 / Qwen3.5 **必须** transformers v5。当前 PyPI 上 `transformers 5.2/5.3` **依赖** `huggingface-hub>=1.3,<2`，因此：

- ✅ `transformers>=5.2,<5.4` + `huggingface-hub>=1.3,<2`
- ❌ 不要再钉 `huggingface-hub<1.0`（与 5.2/5.3 无解；那是旧笔记）

```bash
uv sync --extra train
# 若曾手工钉过 hub<1，先放开再装：
uv pip install "transformers>=5.2.0,<5.4" "huggingface-hub>=1.3.0,<2.0"
uv pip install "huggingface_hub[cli]>=1.3.0,<2.0"
```

自检：

```bash
uv run python -c "import transformers, huggingface_hub; print(transformers.__version__, huggingface_hub.__version__)"
# 期望：transformers 5.2/5.3.x ，hub 1.3+（不是 0.3x）
```

冒烟（应打印 `Qwen3_5ForConditionalGeneration`）：

```bash
uv run huggingface-cli download ATH-MaaS/OvisOCR2
uv run python train/unsloth_stage_a.py --smoke-load-only --model ATH-MaaS/OvisOCR2
```

---

## 4. 数据（合成臂，无 LLM）

```bash
# 渲染模板
uv run python data/scripts/build_synth_batch.py \
  --out data/processed/synth --noise --seed 0

# 扩到约 5000（角落采样 + 噪声多轮）
uv run python data/scripts/expand_synth_to_n.py --target 5000

# 90/5/5
uv run python data/scripts/split_train_val_test.py \
  --input data/processed/synth/point_sharegpt.jsonl \
  --out-dir data/splits
# → train≈4500 / val≈250 / test≈250
```

要点：

- 标签来自 HTML 同源 Markdown，不是 OCR 二次标注  
- bbox 用文字墨迹框（`Range.getClientRects`），准星分层采角落/行首尾  
- `marked/*.jpg` 约 **1.4GB/5k**，**不要进 Git**；云端现造  

磁盘粗算：数据 ~1.5GB + 模型缓存 + checkpoint。

---

## 5. 首次全量训练（保守配方）

小跑验证过（64 samples / 20 steps，loss 明显下降）后再全量。

### 5.1 推荐启动命令（参数写死）

> **RAM 注意：** 旧版脚本会在开训前把全部 JPEG `convert("RGB")` 进内存，~4500 张桌面图可轻易吃光主机内存，而 GPU 显存仍不高。请使用已改为 **lazy 读图** 的 `train/unsloth_stage_a.py`（只在 `__getitem__` 时解码），并加 `--max-image-side 1536`。

```bash
source scripts/setup_autodl_mirrors.sh
export HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com

# 默认写入 checkpoints/<YYYYMMDD_HHMMSS>_stage_a/
uv run python train/unsloth_stage_a.py \
  --data data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --model ATH-MaaS/OvisOCR2 \
  --max-seq-length 4096 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-5 \
  --epochs 1 \
  --max-steps -1 \
  --max-samples 0 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --save-steps 100 \
  --eval-steps 50 \
  --seed 3407 \
  --max-image-side 1536 \
  --num-workers 0
```

含义（保守）：

| 参数 | 值 | 意图 |
|------|-----|------|
| epochs | 1 | 首轮短、先验证效果 |
| lr | 5e-5 | 低于常用 1e-4，更稳 |
| lora-rank / alpha | 8 / 16 | 容量小、过拟合风险低 |
| max-seq-length | 4096 | 比 8192 更快更省显存 |
| grad-accum | 8 | 有效 batch≈8 |
| finetune-vision | 关（默认） | Stage A 先只适配语言侧 |
| eval-steps | 50 | 中途只记 eval/loss；块命中等训后算 |

无 tmux 时：

```bash
nohup uv run python train/unsloth_stage_a.py ... > train_full.log 2>&1 &
tail -f train_full.log
# 或: apt-get install -y tmux && tmux new -s train
```

产出目录（示例）：`checkpoints/20260912_001900_stage_a/`

- `run_config.json` — 全部超参  
- `tb/` — 给 AutoDL TensorBoard  
- `checkpoint-*` + `adapter_final/`  
- `metrics/loss_curves.png`、`final_report.json`（含按文档类型分桶）

### 5.2 小跑 sanity（可选）

```bash
uv run python train/unsloth_stage_a.py \
  --data data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --out checkpoints/smoke_run \
  --max-samples 64 --max-val-samples 16 --max-steps 20 \
  --max-seq-length 4096 --batch-size 1 --grad-accum 8 \
  --lr 5e-5 --lora-rank 8 --lora-alpha 16 --epochs 1 \
  --post-eval-max 16
```

本次实测：loss ~2.6 → ~0.39，流程 OK。

---

## 6. 训后检查（下次继续）

1. AutoDL TensorBoard 指向本次 `.../tb/`；看 train/eval loss  
2. 打开 `metrics/final_report.json` / 曲线图；notebook 目视 5 例（加载 `adapter_final`）  
3. 不够再：新开一轮 run（时间戳目录不会覆盖旧实验） / 略提 `lr` 或 `lora-rank`  

合并（可选）：

```bash
uv run python export/merge_lora.py \
  --base ATH-MaaS/OvisOCR2 \
  --adapter checkpoints/<RUN_ID>/adapter_final \
  --out exports/OvisOCR2-Point-hf
```

---

## 7. 踩坑速查（本轮真实遇到）

| 现象 | 原因 | 处理 |
|------|------|------|
| `curl 16 Error in the HTTP2 framing layer` | AutoDL↔GitHub HTTP/2 | `git -c http.version=HTTP/1.1` + 镜像代理 |
| `libgbm.so.1` / missing dependencies / 造数后无 `point_sharegpt.jsonl` | Chromium 缺系统库，渲染未成功 | `uv run playwright install-deps chromium`（本机实测有效） |
| `huggingface-hub` 与 `transformers` 冲突 | 旧笔记钉 `hub<1`，但 5.2/5.3 要 `hub>=1.3` | 改用 `hub>=1.3,<2` + `transformers>=5.2,<5.4` |
| `model type qwen3_5` / 要求 `transformers>=5.2` | 装了 4.57 | 升到 `transformers>=5.2,<5.4` |
| Xet `401 Unauthorized` 后 retry | HF Xet 传输不稳 | `export HF_HUB_DISABLE_XET=1`，走镜像 |
| uv 暂时装不上 | 网络 | 可用 `python -m venv` + pip；uv 恢复后仍建议回 uv |

---

## 8. 复用清单（新任务复制这套）

1. `source scripts/setup_autodl_mirrors.sh` + `HF_HUB_DISABLE_XET=1`  
2. `uv sync` 对应 extras + `playwright install` + **`install-deps`**  
3. 钉 **transformers 5.2–5.3** + **huggingface-hub 1.3+** + 最新 unsloth  
4. `--smoke-load-only` 看到正确 `ForConditionalGeneration` 类再训  
5. 数据：build → expand → split（大图不进 Git）  
6. 首轮：**1 epoch + 小 LoRA + 中等 seq + 冻 vision**  
7. 再谈效果与加训，不在首轮上猛参  

---

## 9. 相关文件

- 镜像：`scripts/setup_autodl_mirrors.sh`  
- 训练入口：`train/unsloth_stage_a.py` / `train/run_unsloth.sh`  
- 操作手册：`docs/AUTODL.md`  
- Unsloth 说明：`docs/UNSLOTH.md`  
- 一站式 notebook：`notebooks/PointOCR_Full_Pipeline.ipynb`  

（效果评估与后续改进方案：待本次 1-epoch 跑完后再补一节。）

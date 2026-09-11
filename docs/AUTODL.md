# AutoDL 简明操作手册（从开机到训练）

面向本仓库 `point-conditioned-ocr-finetuning`。假设：Ubuntu + NVIDIA GPU（建议 24GB，如 4090）、已创建实例并打开 JupyterLab / 终端。

更完整的框架说明见 `docs/FRAMEWORK.md`；本页只保留**要敲的命令**。

**AutoDL 弱网 + Unsloth + OvisOCR2（Qwen3.5）可复用配方：** [`docs/AUTODL_UNSLOTH_QWEN35_RECIPE.md`](AUTODL_UNSLOTH_QWEN35_RECIPE.md)

---

## 0. 弱网镜像（每个新终端先做）

AutoDL 直连 GitHub / PyPI / Hugging Face / Playwright CDN 经常失败。进仓库后**先**：

```bash
cd Point-conditioned-OCR-finetuning   # 你的实际路径
source scripts/setup_autodl_mirrors.sh
```

会配置：

| 用途 | 环境变量 / 镜像 |
|------|-----------------|
| uv / pip | 阿里云 + 清华 PyPI |
| Hugging Face 模型 | `HF_ENDPOINT=https://hf-mirror.com` |
| Playwright 浏览器 | npmmirror playwright |
| GitHub clone 提示 | `ghproxy.net` |

也可写入 `~/.bashrc`，开机自动生效：

```bash
echo 'source /root/Point-conditioned-OCR-finetuning/scripts/setup_autodl_mirrors.sh' >> ~/.bashrc
```

（路径改成你的仓库绝对路径。）

---

## 为什么要 `playwright install chromium`？

合成数据管线用 **Playwright** 把 HTML/CSS **真的画成像素图**，并在同一页面里跑 JS 读取每个块的文字墨迹框（bbox）。

`chromium` 就是这套无头浏览器引擎：

- **不是**给模型训练用的（训练用的是 GPU + Unsloth（或备选 LLaMA-Factory））  
- **是**「HTML → 截图 + DOM 几何」的渲染器  

不装 Chromium，`build_synth_batch.py` 无法出图。若只下载现成 `marked/` 图片、不重新渲染，可以不装。

任选其一：

```bash
# A. git（弱网）
source scripts/setup_autodl_mirrors.sh   # 若已 clone 可跳过
git -c http.version=HTTP/1.1 clone --depth 1 \
  ${GITHUB_PROXY}/https://github.com/DerekChin12138/Point-conditioned-OCR-finetuning.git

# B. 本地打包后上传（不要带 .venv）
# 本地 Mac:
#   tar --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
#       -czf point-ocr.tgz "point-conditioned-ocr-finetuning"
# AutoDL 终端解压:
#   tar -xzf point-ocr.tgz && cd point-conditioned-ocr-finetuning
```

---

## 1. 安装 uv 与项目环境

```bash
# uv 安装脚本若很慢，可本机装好 uv 二进制，或：
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd /root/Point-conditioned-OCR-finetuning   # 改成你的实际路径
source scripts/setup_autodl_mirrors.sh

uv sync --extra dev --extra synth
uv run playwright install chromium
# 若缺系统库：
# uv run playwright install-deps
```

检查 GPU：

```bash
nvidia-smi
```

（可选）Jupyter kernel：

```bash
uv pip install jupyter ipykernel
uv run python -m ipykernel install --user --name=point-ocr --display-name="point-ocr"
# JupyterLab 里打开一站式流程:
#   notebooks/PointOCR_Full_Pipeline.ipynb
```

---

## 2. 生成有监督数据（合成臂，无需自己截图）

```bash
# 全量模板 → JSONL + 带准星图片
uv run python data/scripts/build_synth_batch.py \
  --out data/processed/synth \
  --noise \
  --seed 0 \
  --r-min 3 --r-max 5

# 划分 train/val/test（90/5/5）
uv run python data/scripts/split_train_val_test.py \
  --input data/processed/synth/point_sharegpt.jsonl \
  --out-dir data/splits
```

产出：

- `data/processed/synth/marked/*.jpg` — 带准星图  
- `data/processed/synth/point_sharegpt.jsonl` — 全量监督对  
- `data/splits/{train,val,test}.jsonl` — 划分后集合  

**推荐在 Jupyter 里按块跑全流程：** `notebooks/PointOCR_Full_Pipeline.ipynb`

抽检：随机打开 `marked/`，确认准星落在文字上（含行首/行尾/角落，而不只是正中心）。

扩到约 5000 条（**不调用 LLM**；复用已渲染页 + 角落采样 + 噪声）：

```bash
uv run python data/scripts/expand_synth_to_n.py --target 5000
uv run python data/scripts/split_train_val_test.py \
  --input data/processed/synth/point_sharegpt.jsonl \
  --out-dir data/splits
# → train 4500 / val 250 / test 250（默认 90/5/5）
```

> 带准星 JPEG 体积大（约 1.4GB / 5k），**不要**提交进 Git；云端 clone 代码后本地再造数。

---

## 3. 安装训练栈（推荐 Unsloth）

```bash
cd /root/Point-conditioned-OCR-finetuning
source scripts/setup_autodl_mirrors.sh   # 确保 HF / PyPI 镜像仍在

uv sync --extra train
uv pip install unsloth

# 冒烟：OvisOCR2 能否被 FastVisionModel 加载
# 模型会走 HF_ENDPOINT（hf-mirror）
uv run python train/unsloth_stage_a.py --smoke-load-only --model ATH-MaaS/OvisOCR2
```

详情与失败回退：[`docs/UNSLOTH.md`](UNSLOTH.md)

Hugging Face 拉基座（已设 `HF_ENDPOINT` 时自动走镜像）：

```bash
huggingface-cli download ATH-MaaS/OvisOCR2
# 如需 token: huggingface-cli login
```

### 备选：LLaMA-Factory

仅当 Unsloth 无法加载 OvisOCR2 时使用：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git /root/LLaMA-Factory
# 按其 README 安装
export LLAMA_FACTORY_ROOT=/root/LLaMA-Factory
# 另需 merge 到 data/llamafactory/ 后再:
# bash train/run_train.sh stage_a
```

---

## 4. 开训 Stage A（Unsloth）

```bash
cd /root/point-conditioned-ocr-finetuning
tmux new -s train
bash train/run_unsloth.sh
# 调试:
# uv run python train/unsloth_stage_a.py --max-samples 64 --max-steps 20
```

权重默认：`checkpoints/stage_a_point_unsloth/lora_adapter`。

---

## 5. 评测

```bash
# 建 held-out（正式请换成未参训页；冒烟可用现模板）
uv run python eval/build_heldout.py --out eval/heldout

# 装推理依赖（与驱动匹配；版本敏感）
uv sync --extra eval

# 零样本基线
uv run python eval/baselines/zero_shot_ovis.py

# 微调后（模型路径改成你的 merge 目录或可被 vLLM 加载的路径）
uv run python eval/run_eval.py --backend vllm \
  --model ATH-MaaS/OvisOCR2 \
  --out eval/results/report.json \
  --pred-out eval/results/pred.jsonl
```

看：`block_hit_rate` ↑、`over_extraction_rate` ↓、`empty_on_chrome_rate` ↑。

---

## 6. 导出 GGUF

```bash
uv run python export/merge_lora.py \
  --base ATH-MaaS/OvisOCR2 \
  --adapter checkpoints/stage_a_point_unsloth/lora_adapter \
  --out exports/OvisOCR2-Point-hf

# 另需 clone 并编译 llama.cpp
export LLAMA_CPP_ROOT=/root/llama.cpp
bash export/export_gguf.sh \
  exports/OvisOCR2-Point-hf \
  exports/OvisOCR2-Point-Q4_K_M.gguf \
  Q4_K_M
```

---

## 7. 推荐节奏（AutoDL 省钱）

1. 小数据冒烟造数 + 目视准星（可不占长时间 GPU）  
2. 短训 0.5–1 epoch 看是否不再整页倾倒  
3. 再放量模板 / 加噪声重训  
4. 达标再租机做 export  

数据扩容：往 `data/synth/templates/` 加更多带 `data-block-id` 的 HTML，然后重跑第 2 步。

---

## 8. 常见问题

| 问题 | 处理 |
|------|------|
| Playwright 失败 | `playwright install chromium` / `install-deps` |
| 图片路径找不到 | 在**同一台机器**上造数；勿混用本机绝对路径 |
| CUDA OOM | 保持 `freeze_vision_tower: true`；开 `quantization_bit: 4`；减小 `cutoff_len` |
| template 报错 | 改 YAML `template` 对齐模型 |
| 仍整页倾倒 | 加负例、查标签是否全文、确认准星够大 |

本地预览样例目录：`data/preview_samples/`（若已生成）。

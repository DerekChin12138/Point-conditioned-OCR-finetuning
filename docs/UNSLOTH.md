# 训练框架：Unsloth（推荐）vs LLaMA-Factory

## 结论

**可以改成 Unsloth，本仓已把 Unsloth 作为 Stage A 推荐入口。**

| | Unsloth | LLaMA-Factory |
|--|---------|----------------|
| 安装 | 单包为主，AutoDL 友好 | 需另 clone 大仓库 |
| VL 微调 | `FastVisionModel` + `UnslothVisionDataCollator` | YAML + template 名 |
| 显存 | 4bit + 自研优化，通常更省 | 视配置 |
| OvisOCR2 | **需上机冒烟**（非官方精调清单里的常见名） | 同样需核对 template |
| 导出 GGUF | Unsloth 对 Qwen3-VL 系较好；Ovis 需实测 | 走 llama.cpp 转换 |

数据协议不变：仍是「带准星图 + POINT 提示 → 单块 Markdown」。只换训练启动器。

---

## AutoDL 安装（CUDA）

```bash
cd /root/point-conditioned-ocr-finetuning

# 基础 + 合成（若还要造数）
uv sync --extra dev --extra synth

# Unsloth：建议直接 pip/uv 装官方包（版本随 CUDA/torch 变化，以官网为准）
uv pip install unsloth
# 若冲突，按其文档用：
# pip install "unsloth[cu128]"  # 示例，按你机器 CUDA 版本选

# 冒烟：能否加载 OvisOCR2
uv run python train/unsloth_stage_a.py --smoke-load-only --model ATH-MaaS/OvisOCR2
```

### 若 `--smoke-load-only` 失败

常见原因：Unsloth 尚未对该 checkpoint 的架构做特化 / `trust_remote_code` 行为差异。

可选：

1. **升级** `unsloth` / `transformers` 后再试  
2. 用 Unsloth 官方支持的接近基座做对照实验（会偏离任务书基座）：  
   `unsloth/Qwen3-VL-2B-Instruct` 等 —— 仅作管线验证，正式交付仍优先 OvisOCR2  
3. 临时回退 `train/stage_a_point_qlora.yaml`（LLaMA-Factory）

---

## 开训

```bash
# 先有数据
uv run python data/scripts/build_synth_batch.py --out data/processed/synth --noise

tmux new -s train
bash train/run_unsloth.sh
# 等价：
# uv run python train/unsloth_stage_a.py \
#   --data data/processed/synth/point_sharegpt.jsonl \
#   --out checkpoints/stage_a_point_unsloth

# 调试小跑
uv run python train/unsloth_stage_a.py --max-samples 64 --max-steps 20
```

默认：**冻 vision**（不加 `--finetune-vision`），LoRA 语言侧，对齐 Stage A 设定。

产出：`checkpoints/stage_a_point_unsloth/lora_adapter/`

合并仍可用：

```bash
uv run python export/merge_lora.py \
  --base ATH-MaaS/OvisOCR2 \
  --adapter checkpoints/stage_a_point_unsloth/lora_adapter \
  --out exports/OvisOCR2-Point-hf
```

---

## 数据格式（Unsloth）

训练脚本会把本仓 ShareGPT JSONL 转成：

```python
{"messages": [
  {"role": "user", "content": [
    {"type": "text", "text": POINT_PROMPT},
    {"type": "image", "image": <PIL.Image>}
  ]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "<block markdown or empty>"}
  ]}
]}
```

无需再依赖 `data/llamafactory/`（该目录仅 LLaMA-Factory 备选需要）。

---

## LLaMA-Factory（备选）

保留 `train/stage_a_point_qlora.yaml` + `train/run_train.sh`。仅当 Unsloth 无法加载 OvisOCR2 时使用。

# Unsloth 安装（CUDA）

训练框架只走 **Unsloth** `FastVisionModel` + TRL。启动命令与超参见仓库根 [`README.md`](../README.md)。

```bash
uv sync --extra train
uv pip install unsloth
uv run python train/unsloth_stage_a.py --smoke-load-only --model models/Qwen3.5-0.8B
```

OvisOCR2 / Qwen3.5 需要 `transformers>=5.2,<5.4`（已写在 `pyproject.toml` 的 `train` extra）。弱网镜像见 [`AUTODL.md`](AUTODL.md)。

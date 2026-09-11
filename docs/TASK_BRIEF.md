# 任务书：OvisOCR2 点条件语义块 OCR 微调

把本文当作新 Agent / 新仓库的完整提示词。训练仓已按此落地骨架；继续扩数据规模、在 CUDA 主机上训练与导出。

**必读参考论文：** [OvisOCR2 Technical Report (arXiv:2607.13639)](https://arxiv.org/abs/2607.13639) · [HTML 版](https://arxiv.org/html/2607.13639v1) · 模型卡 [ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2)

---

## 1. 背景（简要）

OvisOCR2 是基于 Qwen3.5-0.8B 的端到端文档解析模型：输入整页图像，输出整页 Markdown。它学会了「画面里有语义块、如何串成阅读顺序」，但**没有**「给定图像上的一个兴趣点，只输出包含该点的那一块」这一条件任务。实践表明：仅改提示词 + 在图上画准星，零样本几乎仍会倾倒整页内容。因此需要**专门微调**。

产品侧最终希望：悬停/热键时在截图上画准星，模型只 OCR 准星附近的最小语义块，再交给翻译。训练与产品解耦——交付可用的点条件权重（及可选的双任务权重）。

## 2. 微调任务目标

### 2.1 主任务 POINT（必须做好）

| 项 | 定义 |
|----|------|
| 输入 | 文档页或屏幕截图像素图 + 品红/白十字准星与圆环 + 固定 POINT 指令 |
| 输出 | **仅一个**最小语义块的 Markdown；不是整页 |
| 负例 | 准星在空白/图标/无字 chrome → 空字符串 |
| 失败模式 | 倾倒整页、吐出邻块/另一栏、忽略准星 |

### 2.2 可选 PAGE + POINT（双任务）

Stage A 达标后再试同一权重用提示词切换；互相干扰则保留独立 POINT 权重。

### 2.3 分阶段交付

1. **Stage A**：独立 `OvisOCR2-Point` + GGUF（及 mmproj）  
2. **Stage B（可选）**：双任务混训  

算力：单卡约 24GB NVIDIA，Windows 或 Linux。默认不做 4B GRPO + OPD。

## 3–6

详见仓库根 `README.md` 与 `docs/STAGES.md`、`docs/MARKER_SPEC.md`、`docs/PROMPTS.md`。实现入口：

- 准星：`src/point_ocr/marker.py`  
- 数据：`data/scripts/build_synth_batch.py`、`build_real_batch.py`  
- 训练：`train/stage_a_point_qlora.yaml`  
- 评测：`eval/run_eval.py`  
- 导出：`export/`  

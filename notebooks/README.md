# Notebooks（云端推荐入口）

**一站式全流程：** [`PointOCR_Full_Pipeline.ipynb`](PointOCR_Full_Pipeline.ipynb)（推荐）


按序号执行：

| Notebook | 内容 |
|----------|------|
| `00_cloud_setup.ipynb` | uv / CUDA / Jupyter kernel / extras |
| `01_marker_demo.ipynb` | 准星与提示词可视化 |
| `02_build_synth_data.ipynb` | 合成 POINT 数据 |
| `03_merge_for_llamafactory.ipynb` | 合并训练 JSON |
| `04_train_stage_a.ipynb` | Unsloth Stage A 开训 |
| `05_eval.ipynb` | held-out 指标 |
| `06_export.ipynb` | merge + GGUF |

完整文字说明：[`docs/FRAMEWORK.md`](../docs/FRAMEWORK.md)

底层 `.py` 仍保留；notebook 逐步调用它们，方便在租卡环境点跑与改参数。

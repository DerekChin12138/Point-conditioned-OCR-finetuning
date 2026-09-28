# Qwen3.5-2B 迁移可行性实测（RTX 4060 Laptop 8GB）

Smoke: `train/unsloth_stage_a.py` on `data/splits_stage_q2_ocr_mt_v2/train.jsonl`,
batch/grad-accum as noted, max_pixels=8294400, LoRA r32/α64, frozen vision.

| model | precision | max_seq | batch×accum | peak GPU (MiB) | samples | train_runtime | s/sample |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 4bit | 4096 | 2×8 | 1969 | 64 | — | ~0.36* |
| Qwen3.5-0.8B | 4bit | 8192 | 2×8 | 1865 | 64 | 45.35 | 0.71 |
| Qwen3.5-2B | 4bit | 4096 | 2×8 | 3311 | 96 | 116.7 | 1.22 |
| Qwen3.5-2B | 4bit | 8192 | 2×8 | 3893 | 48 | 59.77 | 1.25 |
| Qwen3.5-2B | bf16 | 8192 | 1×16 | 5205 | 48 | 60.63 | 1.26 |
| Qwen3.5-2B | bf16 | 4096 | 1×16 | (fit) | 32 | 33.99 | 1.06 |

*0.8B 4096 run hit the 4096 token cap; the 8192 row is the fair comparison.

Config: `qwen3.5-2b.config.json`.
Key facts: same `Qwen3_5ForConditionalGeneration`, same vocab/token ids/processor;
text hidden 2048 (vs 1024), vision depth 24×1024 (vs 12×768).
Densest pages reach ~8000 vision+text tokens → `max_seq_length=4096` truncates ~10%.

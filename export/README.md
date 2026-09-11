# Export: LoRA merge → GGUF (+ mmproj)

## Stage A deliverable

1. Merge adapter: `python export/merge_lora.py --adapter checkpoints/stage_a_point_qlora --out exports/OvisOCR2-Point-hf`
2. Quantize: `./export/export_gguf.sh exports/OvisOCR2-Point-hf exports/OvisOCR2-Point-Q4_K_M.gguf Q4_K_M`
3. Also produce `Q5_K_M` if quality drop is visible on held-out cells/formulas.
4. If the deployment stack requires a separate multimodal projector GGUF (`mmproj`), convert the vision tower with the same `llama.cpp` revision you use at inference and ship it next to the text GGUF.

## Notes

- Export on **Linux/Windows + CUDA** (or CPU convert); not macOS-only MLX.
- Verify POINT prompt + `MARKER_SPEC` crosshair at inference — train/serve must match.
- Optional: average 2–3 Stage A LoRA runs (different seeds/LR) via weighted parameter fusion before merge, as suggested by the OvisOCR2 report, if you have multiple configs.

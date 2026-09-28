# Prompt contract

Source of truth: `src/point_ocr/prompts.py`.

## POINT (Stage A / Q1)

Static instruction (identical for every sample). Q1 default is **`a2_v3`**:

- Magenta/white 45° X is on the full image
- Output **one** semantic block most related to the mark (or a tightly bound short group)
- Empty string if blank / chrome / decoration / table / image-or-chart
- Formulas and code are transcribed as Markdown (not emptied)
- Do not output the full page or neighbors

Training occupancy gate: page ink union in **[25%, 70%]**.

`a2_v2` stays available for Ovis B1 jsonl. `POINT_PROMPT` still aliases `a2_v3`. Use `get_prompt("POINT", prompt_key=…)` — do not hand-write variants.

## POINT + MT (Q2)

**`ocr_mt_v1`** (`POINT_PROMPT_OCR_MT_V1`): same geometry and empty rule as `a2_v3`. Positives must be exactly:

```xml
<source>
original Markdown
</source>
<translation>
Simplified Chinese
</translation>
```

Empty GT stays `""` (no empty XML tags). Formulas/code go inside `<source>`, then translated. Implemented in `src/point_ocr/ocr_mt.py`.

## PAGE (Stage B only)

Aligned with the official OvisOCR2 page Markdown instruction from the [model card](https://huggingface.co/ATH-MaaS/OvisOCR2) (full-page reading order, formulas LaTeX, tables HTML, visual regions as bbox `<img>` tags).

## Dual-task switching

Same weights, different user text. Do not mix task semantics inside one sample. If interference appears, ship POINT-only.

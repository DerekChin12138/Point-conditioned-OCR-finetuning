# Prompt contract

Source of truth: `src/point_ocr/prompts.py`.

## POINT (Stage A required)

Emphasizes:

- Magenta/white crosshair is present on the full image  
- Output **one** minimal semantic block containing the point  
- Forbid full-page dump / neighbors / other columns  
- Empty string if blank / icon / chrome  

## PAGE (Stage B only)

Aligned with the official OvisOCR2 page Markdown instruction from the [model card](https://huggingface.co/ATH-MaaS/OvisOCR2) (full-page reading order, formulas LaTeX, tables HTML, visual regions as bbox `<img>` tags).

## Dual-task switching

Same weights, different user text. Do not mix task semantics inside one sample. If interference appears, ship POINT-only.

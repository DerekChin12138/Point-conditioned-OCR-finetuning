"""Fixed English instruction prompts for POINT and PAGE tasks.

Keep these strings identical between training data builders and inference.
Style aligned with OvisOCR2's page-level Markdown instruction.
"""

from __future__ import annotations

# Official-style PAGE prompt (from ATH-MaaS/OvisOCR2 model card), Stage B only.
PAGE_PROMPT = (
    "Extract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text "
    "without translation or paraphrasing."
)

# POINT: single minimal semantic block under the crosshair.
POINT_PROMPT = (
    "The image contains a prominent magenta-and-white crosshair (interest point). "
    "Identify the single minimal semantic block whose region contains that crosshair "
    "(e.g. one paragraph, heading, list item, table cell, formula, or short code span). "
    "Output ONLY that block as Markdown (formulas as LaTeX; table cells as plain text "
    "or a minimal HTML <td> fragment when needed). "
    "Do NOT dump the full page, neighboring blocks, or other columns. "
    "If the crosshair lies on blank space, icons, chrome, or any region without readable text, "
    "output an empty string."
)

TASK_PROMPTS = {
    "POINT": POINT_PROMPT,
    "PAGE": PAGE_PROMPT,
}


def get_prompt(task: str) -> str:
    key = task.strip().upper()
    if key not in TASK_PROMPTS:
        raise KeyError(f"Unknown task {task!r}; expected one of {sorted(TASK_PROMPTS)}")
    return TASK_PROMPTS[key]

"""Fixed English instruction prompts for POINT and PAGE tasks.

Keep these strings identical between training data builders and inference.
Style aligned with OvisOCR2's page-level Markdown instruction.
"""

from __future__ import annotations

import json

# Official PAGE prompt (from ATH-MaaS/OvisOCR2 model card) — Stage B / base model.
PAGE_PROMPT = (
    "Extract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text "
    "without translation or paraphrasing."
)

# A1_v2: minimal single block only (no short-group expansion).
POINT_PROMPT_A1_V2 = (
    "The image contains a prominent magenta-and-white cross rotated 45 degrees "
    "(an X-shaped interest point). "
    "Identify the single minimal semantic block under that mark "
    "(e.g. one paragraph, heading, or list item). "
    "Output ONLY that block as Markdown. "
    "If the mark is on blank space, chrome, decoration, or non-text, output an empty string. "
    "Do NOT dump the full page, other columns, or neighboring blocks."
)

# A2_v2: allows short-group units (seed + tightly bound neighbors).
POINT_PROMPT_A2_V2 = (
    "The image contains a prominent magenta-and-white cross rotated 45 degrees "
    "(an X-shaped interest point). "
    "Identify the single semantic unit under that mark "
    "(one text block, or a short anchor with its tightly bound nearby group). "
    "Output ONLY that unit as Markdown. Format tables as HTML <table>...</table>. "
    "If the mark is on blank space, chrome, decoration, formula, code, or image/chart, "
    "output an empty string. "
    "Do NOT dump the full page, other columns, or text outside the unit."
)

# A2_v3 / Q1: empty only on blank / chrome / table / image (not formula or code).
POINT_PROMPT_A2_V3 = (
    "The image contains a magenta-and-white cross rotated 45 degrees "
    "(an X-shaped interest point). "
    "Identify the single semantic block most related to that mark "
    "(one text block, or a short anchor with its tightly bound nearby group). "
    "Output ONLY that unit as Markdown. Transcribe formulas and code as Markdown. "
    "If the mark is on blank space, chrome, decoration, a table, or an image/chart, "
    "output an empty string. "
    "Do NOT output the full page, other columns, or text outside the unit."
)

# Q2 OCR + English→Chinese: same POINT geometry as a2_v3; positives are XML.
POINT_PROMPT_OCR_MT_V1 = (
    "The image contains a magenta-and-white cross rotated 45 degrees "
    "(an X-shaped interest point). "
    "Identify the single semantic block most related to that mark "
    "(one text block, or a short anchor with its tightly bound nearby group). "
    "If the mark is on readable text, output ONLY that unit in this exact form:\n"
    "<source>\n"
    "the original text as Markdown\n"
    "</source>\n"
    "<translation>\n"
    "a faithful Simplified Chinese translation of that source\n"
    "</translation>\n"
    "Transcribe formulas and code as Markdown inside <source>, then translate them. "
    "If the mark is on blank space, chrome, decoration, a table, or an image/chart, "
    "output an empty string. "
    "Do NOT output the full page, other columns, or text outside the unit. "
    "Do NOT add commentary outside the two tags."
)

# Default POINT prompt = A2_v3 (Qwen Q1 / current studio).
POINT_PROMPT = POINT_PROMPT_A2_V3

# q1_coordinate: unmarked image + native pixel point (Qwen point_2d).
# Origin = top-left, x right, y down. `{point}` → [{"point_2d": [x, y]}].
POINT_PROMPT_COORD_Q1_TEMPLATE = (
    "A query interest point is given as image pixel coordinates "
    "in a JSON list with key point_2d [x, y]. "
    "Origin is the top-left corner (x increases right, y increases downward).\n"
    "{point}\n"
    "Identify the single semantic block most related to that point "
    "(one text block, or a short anchor with its tightly bound nearby group). "
    "Output ONLY that unit as Markdown. Transcribe formulas and code as Markdown. "
    "If the point is on blank space, chrome, decoration, a table, or an image/chart, "
    "output an empty string. "
    "Do NOT output the full page, other columns, or text outside the unit."
)

# Studio color-prior probe: deliberately does NOT name the origin.
POINT_PROMPT_COORD_PROBE_TEMPLATE = (
    "The image has four solid-color quadrants and no other content.\n"
    "{point}\n"
    "What color is at that point? Reply with one word: red, green, blue, or yellow."
)

POINT_PROMPT_RED_REGION = (
    "The image contains four solid-color quadrants and nothing else. "
    "Precisely describe the spatial extent of the RED region only. "
    "Use whatever coordinate system or notation you prefer "
    "(for example pixels, normalized numbers, percentages, a bounding box, or a markup tag). "
    "Be as specific as you can about the region's boundaries. "
    "Do not describe the other colors."
)

# Legacy integer-grid draft (do not use for q1_coordinate).
POINT_PROMPT_COORD_V1_TEMPLATE = (
    "The image has an interest point at normalized coordinates "
    "x={x}, y={y} on a [0, 1000) grid "
    "(origin at the top-left; x increases right, y increases down). "
    "No mark is drawn on the image. "
    "Identify the single semantic block most related to that point "
    "(one text block, or a short anchor with its tightly bound nearby group). "
    "Output ONLY that unit as Markdown. Transcribe formulas and code as Markdown. "
    "If the point is on blank space, chrome, decoration, a table, or an image/chart, "
    "output an empty string. "
    "Do NOT output the full page, other columns, or text outside the unit."
)

# Frozen legacy prompts (old checkpoints only).
POINT_PROMPT_A1 = (
    "The image contains a prominent magenta-and-white crosshair (interest point). "
    "Identify the single minimal semantic block whose region contains that crosshair "
    "(e.g. one paragraph, heading, list item, table cell, formula, or short code span). "
    "Output ONLY that block as Markdown (formulas as LaTeX; table cells as plain text "
    "or a minimal HTML <td> fragment when needed). "
    "Do NOT dump the full page, neighboring blocks, or other columns. "
    "If the crosshair lies on blank space, icons, chrome, or any region without readable text, "
    "output an empty string."
)

POINT_PROMPT_A2_STAR = (
    "The image contains a prominent magenta dotted four-pointed star (interest point). "
    "Identify the single semantic unit under that mark "
    "(one block, or a short anchor with its tightly bound nearby context). "
    "Output ONLY that unit as Markdown. Format tables as HTML <table>...</table>. "
    "If the mark is on a formula, code, image/chart, blank space, icons, or chrome, "
    "output an empty string. "
    "Do NOT dump the full page or unrelated columns."
)

TASK_PROMPTS = {
    "POINT": POINT_PROMPT_A2_V3,
    "PAGE": PAGE_PROMPT,
    "POINT_A1_V2": POINT_PROMPT_A1_V2,
    "POINT_A2_V2": POINT_PROMPT_A2_V2,
    "POINT_A2_V3": POINT_PROMPT_A2_V3,
    "POINT_OCR_MT_V1": POINT_PROMPT_OCR_MT_V1,
    "POINT_COORD_Q1": POINT_PROMPT_COORD_Q1_TEMPLATE,
    "POINT_COORD_V1": POINT_PROMPT_COORD_V1_TEMPLATE,
}

POINT_COORD_GRID = 1000


def pixel_to_norm(x: float, y: float, image_w: int, image_h: int) -> tuple[int, int]:
    """Map pixel coords to integer [0, 1000) grid (Ovis-style)."""
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"invalid image size {(image_w, image_h)}")
    x = min(max(float(x), 0.0), float(image_w - 1))
    y = min(max(float(y), 0.0), float(image_h - 1))
    nx = int(x / image_w * POINT_COORD_GRID)
    ny = int(y / image_h * POINT_COORD_GRID)
    nx = min(max(nx, 0), POINT_COORD_GRID - 1)
    ny = min(max(ny, 0), POINT_COORD_GRID - 1)
    return nx, ny


def format_point_prompt(
    x: float,
    y: float,
    image_w: int,
    image_h: int,
) -> str:
    """Legacy coord-augmented prompt (not used by current Stage A protocol)."""
    nx, ny = pixel_to_norm(x, y, image_w, image_h)
    return (
        f"{POINT_PROMPT} "
        f"(Debug coords on [0, 1000) grid: x={nx}, y={ny}.)"
    )


def pixel_to_pct_topleft(
    x: float, y: float, image_w: int, image_h: int
) -> tuple[float, float]:
    """Pixel click → percent of width/height, origin top-left, 1 decimal."""
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"invalid image size {(image_w, image_h)}")
    x = min(max(float(x), 0.0), float(image_w - 1))
    y = min(max(float(y), 0.0), float(image_h - 1))
    xp = round(100.0 * x / float(image_w), 1)
    yp = round(100.0 * y / float(image_h), 1)
    xp = min(max(xp, 0.0), 99.9)
    yp = min(max(yp, 0.0), 99.9)
    return xp, yp


def format_point_tag(x_pct: float, y_pct: float) -> str:
    return f"<point>\nx: {x_pct:.1f}%\ny: {y_pct:.1f}%\n</point>"


def pixel_to_xy_int(
    x: float, y: float, image_w: int, image_h: int
) -> tuple[int, int]:
    """Clamp a click to integer pixels on this image (origin top-left)."""
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"invalid image size {(image_w, image_h)}")
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    xi = min(max(xi, 0), image_w - 1)
    yi = min(max(yi, 0), image_h - 1)
    return xi, yi


def format_point_2d_json(x: float, y: float) -> str:
    """Qwen-style grounding payload: [{"point_2d": [x, y]}]."""
    xi, yi = int(round(float(x))), int(round(float(y)))
    return json.dumps([{"point_2d": [xi, yi]}], ensure_ascii=False)


def format_coord_prompt_q1(
    x: float,
    y: float,
    image_w: int,
    image_h: int,
    *,
    template: str | None = None,
) -> str:
    """Fill q1_coordinate prompt from a pixel click (top-left, integer pixels)."""
    xi, yi = pixel_to_xy_int(x, y, image_w, image_h)
    raw = template if template is not None else POINT_PROMPT_COORD_Q1_TEMPLATE
    return raw.replace("{point}", format_point_2d_json(xi, yi))


def format_coord_prompt_v1(
    x: float,
    y: float,
    image_w: int,
    image_h: int,
    *,
    template: str | None = None,
) -> str:
    """Legacy [0, 1000) integer grid."""
    nx, ny = pixel_to_norm(x, y, image_w, image_h)
    raw = template if template is not None else POINT_PROMPT_COORD_V1_TEMPLATE
    return apply_coord_placeholders(raw, nx, ny)


def apply_coord_placeholders(prompt: str, x_val: float | int, y_val: float | int) -> str:
    """Replace `{point}` / `{x}` / `{y}` if present. Leave other prompts unchanged."""
    if "{" not in prompt:
        return prompt
    if isinstance(x_val, float) or isinstance(y_val, float):
        xs, ys = f"{float(x_val):.1f}", f"{float(y_val):.1f}"
        tag = format_point_tag(float(x_val), float(y_val))
    else:
        xs, ys = str(int(x_val)), str(int(y_val))
        tag = f"<point>\nx: {xs}\ny: {ys}\n</point>"
    return (
        prompt.replace("{point}", tag)
        .replace("{x}", xs)
        .replace("{y}", ys)
        .replace("{nx}", xs)
        .replace("{ny}", ys)
        .replace("{x_norm}", xs)
        .replace("{y_norm}", ys)
    )


def get_prompt(task: str, *, prompt_key: str | None = None) -> str:
    """Resolve user prompt.

    ``prompt_key`` wins when set (``a1_v2`` / ``a2_v2`` / ``a2_v3`` / TASK_PROMPTS key).
    """
    if prompt_key:
        key = str(prompt_key).strip()
        aliases = {
            "a1_v2": "POINT_A1_V2",
            "a2_v2": "POINT_A2_V2",
            "a2_v3": "POINT_A2_V3",
            "ocr_mt": "POINT_OCR_MT_V1",
            "ocr_mt_v1": "POINT_OCR_MT_V1",
            "point_ocr_mt": "POINT_OCR_MT_V1",
            "point_ocr_mt_v1": "POINT_OCR_MT_V1",
            "coord_v1": "POINT_COORD_V1",
            "coord_q1": "POINT_COORD_Q1",
            "q1_coordinate": "POINT_COORD_Q1",
            "point_a1_v2": "POINT_A1_V2",
            "point_a2_v2": "POINT_A2_V2",
            "point_a2_v3": "POINT_A2_V3",
            "point_coord_v1": "POINT_COORD_V1",
            "point_coord_q1": "POINT_COORD_Q1",
        }
        resolved = aliases.get(key.lower(), key)
        if resolved in TASK_PROMPTS:
            return TASK_PROMPTS[resolved]
        if key in TASK_PROMPTS:
            return TASK_PROMPTS[key]
    try:
        return TASK_PROMPTS[task]
    except KeyError as e:
        raise KeyError(f"unknown task {task!r}; expected one of {sorted(TASK_PROMPTS)}") from e

"""Four-quadrant solid color image for coordinate-origin probes."""

from __future__ import annotations

from PIL import Image, ImageDraw

# Pixel origin is top-left (PIL).
QUAD_COLORS = {
    "tl": ((229, 57, 53), "red"),
    "tr": ((67, 160, 71), "green"),
    "bl": ((30, 136, 229), "blue"),
    "br": ((253, 216, 53), "yellow"),
}

PROBE_POINTS = (
    (25.0, 25.0),
    (25.0, 75.0),
    (75.0, 25.0),
    (75.0, 75.0),
)


def make_quadrant_image(size: int = 800) -> Image.Image:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    mid = size // 2
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, mid - 1, mid - 1], fill=QUAD_COLORS["tl"][0])
    draw.rectangle([mid, 0, size - 1, mid - 1], fill=QUAD_COLORS["tr"][0])
    draw.rectangle([0, mid, mid - 1, size - 1], fill=QUAD_COLORS["bl"][0])
    draw.rectangle([mid, mid, size - 1, size - 1], fill=QUAD_COLORS["br"][0])
    draw.line([(mid, 0), (mid, size)], fill=(0, 0, 0), width=4)
    draw.line([(0, mid), (size, mid)], fill=(0, 0, 0), width=4)
    return img


def expected_color(x_pct: float, y_pct: float, *, origin: str) -> str:
    """origin: 'top-left' (y down) or 'bottom-left' (y up)."""
    right = x_pct >= 50.0
    if origin == "bottom-left":
        top = y_pct >= 50.0
    else:
        top = y_pct < 50.0
    if top and not right:
        return QUAD_COLORS["tl"][1]
    if top and right:
        return QUAD_COLORS["tr"][1]
    if not top and not right:
        return QUAD_COLORS["bl"][1]
    return QUAD_COLORS["br"][1]


def interpret_probe(preds: list[dict]) -> str:
    """Compare greedy color words against both origin hypotheses."""
    def hits(origin: str) -> int:
        n = 0
        for row in preds:
            exp = expected_color(row["x_pct"], row["y_pct"], origin=origin)
            got = str(row.get("pred") or "").strip().lower()
            if exp in got:
                n += 1
        return n

    tl, bl = hits("top-left"), hits("bottom-left")
    n = max(len(preds), 1)
    if tl == n and bl < n:
        return (
            f"prior looks TOP-LEFT ({tl}/{n} match). "
            "q1_coordinate percent origin should stay top-left."
        )
    if bl == n and tl < n:
        return (
            f"prior looks BOTTOM-LEFT ({bl}/{n} match). "
            "Flip y in the prompt (origin bottom-left) before SFT."
        )
    if tl == bl:
        return (
            f"ambiguous (top-left {tl}/{n}, bottom-left {bl}/{n}). "
            "Model may be ignoring the point; try base without LoRA."
        )
    return f"mixed (top-left {tl}/{n}, bottom-left {bl}/{n}). Weak spatial prior."

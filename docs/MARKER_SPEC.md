# MARKER_SPEC

Canonical implementation: `src/point_ocr/marker.py` → `MARKER_SPEC`.

## Visual

| Element | Spec |
|---------|------|
| Cross | Magenta `#FF00FF`, half-length 34px, width 5px |
| Cross outline | White, +2px each side |
| Ring | White circle, radius 28px, width 5px |
| Center | Magenta dot, radius 4px |
| Min image | ≥ 48px on short side; marker must stay visible after mild resize |

## Rules

1. Draw on the **full page image** used as model input (not only on a crop).
2. Training images and product inference must call the same drawer / same constants.
3. Do not thin the marker for aesthetics — zero-shot failure modes include “marker disappears after resize”.
4. Interest point `(x, y)` is the geometric center of the crosshair.

## Demo

```bash
point-ocr-marker-demo page.png -o marked.png --x 400 --y 300
# or: python -m point_ocr.cli marker_demo ...
```

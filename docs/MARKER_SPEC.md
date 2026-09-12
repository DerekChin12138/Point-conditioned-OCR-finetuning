# MARKER_SPEC

Canonical implementation: `src/point_ocr/marker.py` → `MARKER_SPEC`.

## Visual (Batch-2)

| Element | Spec |
|---------|------|
| Cross | Magenta `#FF00FF`, half-length **26px**, width **4px** |
| Cross outline | White, +2px each side |
| Center | **Open aperture** — arms stop at `center_gap_radius_px=7`; **no center dot** |
| Ring | White circle, radius **22px**, width **4px** |
| Size policy | **Fixed** footprint (never scaled by bbox; product cannot know block size) |
| Min image | ≥ 40px on short side; marker must stay visible after mild resize |

## Rules

1. Draw on the **full page image** used as model input (not only on a crop).
2. Training images and product inference must call the same drawer / same constants.
3. Do not thin the marker further for aesthetics — zero-shot failure modes include “marker disappears after resize”. Batch-2 already slightly reduces size vs Batch-1; keep train≡serve.
4. Interest point `(x, y)` is the geometric center of the crosshair (inside the clear gap).
5. After changing `MARKER_SPEC`, **rebuild** all marked training/eval images before the next Stage A run. Old checkpoints trained on the previous marker are not comparable.

## Demo

```bash
point-ocr-marker-demo page.png -o marked.png --x 400 --y 300
# or: python -m point_ocr.cli marker_demo ...
```

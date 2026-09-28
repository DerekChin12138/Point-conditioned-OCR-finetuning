# MARKER_SPEC

Canonical implementation: `src/point_ocr/marker.py` → `spec_for_image` + `draw_crosshair`.
Tag: `CURRENT_MARKER_TAG = "x45r"`.

## Role

**Primary** localization signal is a magenta/white cross rotated 45° (an X).
Stage A uses a **static** `POINT_PROMPT` (no per-sample coordinates).

## Size (page-relative, randomized per image)

Treat the X as the two diagonals of a square. **Each marked image** draws its
square/page-area fraction uniformly at random from:

- **current (x45r)** `MARKER_AREA_FRAC_MIN` = **0.0008** … `MARKER_AREA_FRAC_MAX` = **0.002**
  (i.e. the square is **0.08%–0.2%** of the image; `sample_marker_area_frac(rng)`).
- `MARKER_AREA_FRAC` = **0.0014** (midpoint) is only the single-shot default used by
  the demo CLI / studio when no rng is available.

Size variation lives in the **SFT data**, so it is **not** a GRPO objective anymore.

Legacy fixed protocols (old metadata still parses; do not use for new renders):

- **x45c** `MARKER_AREA_FRAC_LEGACY` = **0.005** (0.5%) — Q1 SFT v1
- **x45d** = legacy × **0.85** (= 0.00425) — Q2 SFT v1 default
- **GRPO_2 small** = legacy × **0.5** (only `multi_frag` / `semantic_group`)

On every page, set arm so that the square occupies the sampled fraction:

`half_length = round(sqrt(frac × W × H / 2))`

Stroke widths and ring scale with `half_length / 34`.

> **Note on the numbers.** The legacy baseline was **0.5%**, not 5%. On the
> 2582×1641 reference, x45c is tip-to-tip ≈ 206px (x45d ≈ 190px). The new range
> (0.08%–0.2%) is a **~2.5–5.6× linear shrink** (0.16–0.026× area) of x45c — the
> mark is small and relies on the 2B vision tower for capture. Preview:
> `eval/results/marker_size_preview_zoom.png`.

For legacy dual-size remakes, pass `area_frac=MARKER_AREA_FRAC_LEGACY * {0.85|0.5}`
(do **not** use the linear `scale=` kwarg if you mean an area ratio — it squares the
multiplier).

## Opacity

Fully opaque (**α=255**) on the cross, white outline, ring, and center dot.
Translucent X was unusable in studio: the model lost the cue as soon as alpha dropped.

## Visual (1× template)

| Element | Template |
|---------|----------|
| Cross | Magenta `#FF00FF`, half-length **34px**, width **3px** |
| Rotation | **45°** |
| Outline | White, +1px |
| Ring | White, radius **28px**, width **3px** |
| Center | Magenta dot, radius **3px** |
| Color | Fixed magenta (never adapted to local background) |

## Rules

1. Draw on the **full page image** used as model input — do not crop to the mark.
2. Production callers must use `spec_for_image(w, h)`, never the 34px template alone.
3. Training and product share this drawer + `POINT_PROMPT`.
4. After changing the protocol, **rebuild** all marked datasets before the next train/RL run.

## Demo

```bash
point-ocr-marker-demo page.png -o marked.png --x 400 --y 300
```

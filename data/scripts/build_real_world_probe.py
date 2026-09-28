#!/usr/bin/env python3
"""Stamp current x45 markers onto data/real_world_sample screenshots → ShareGPT JSONL.

Also writes a browser-chrome-cropped variant (tab/address bar removed) so the
目视 cell can A/B full window vs client-area, matching the synth capture contract.

  uv run python data/scripts/build_real_world_probe.py
  # → data/real_world_sample/marked/*.jpg
  # → data/real_world_sample/point_sharegpt.jsonl
  # → data/real_world_sample/nochrome/*.png
  # → data/real_world_sample/marked_nochrome/*.jpg
  # → data/real_world_sample/point_sharegpt_nochrome.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.dataset_format import PointSample, write_jsonl  # noqa: E402
from point_ocr.marker import CURRENT_MARKER_TAG, MARKER_SPEC, draw_crosshair, spec_for_image  # noqa: E402
from point_ocr.prompts import pixel_to_norm  # noqa: E402

SRC = ROOT / "data/real_world_sample"
OUT_MARKED = SRC / "marked"
OUT_JSONL = SRC / "point_sharegpt.jsonl"
OUT_NOCHROME = SRC / "nochrome"
OUT_MARKED_NOCHROME = SRC / "marked_nochrome"
OUT_JSONL_NOCHROME = SRC / "point_sharegpt_nochrome.jsonl"

# These five screenshots share the same Chrome window chrome height
# (title/tab strip + address bar + thin separator ≈ y∈[0,148)).
BROWSER_CHROME_TOP_PX = 148

# (stem, x, y, sample_id, kind, target) — coords in the *full* screenshot.
CASES: list[tuple[str, int, int, str, str, str]] = [
    (
        "0097fb7657aa9131bc2b52cf2745d0fe",
        720,
        328,
        "arxiv_abs__title",
        "core_inner",
        "A Ranking Approach for Measuring Calibration",
    ),
    (
        "0097fb7657aa9131bc2b52cf2745d0fe",
        620,
        560,
        "arxiv_abs__abstract",
        "multi_frag",
        (
            "When providing forecasted probabilities with a predictive model, the ideal "
            "model offers perfect calibration: the true probability of the outcome (i.e., "
            "the probability that $Y = 1$) exactly matches the forecasted probability "
            "$f(X)$. In practice, models inevitably exhibit calibration error, and it is "
            "therefore important to be able to measure this miscalibration to assess a "
            "model's reliability. The Expected Calibration Error (ECE) is the most widely "
            "used measure of miscalibration, but is known to be impossible to estimate the "
            "ECE with guaranteed accuracy in an assumption-free setting. In this work, we "
            "propose an alternative measure, the rankECE, that is based on comparing points "
            "with neighboring values of the predicted probability $f(X)$. Our theoretical "
            "guarantees and empirical results establish that rankECE provides a better proxy "
            "for ECE as compared to binned approximations to ECE, which are the most "
            "commonly-used approximations in practice."
        ),
    ),
    (
        "0097fb7657aa9131bc2b52cf2745d0fe",
        2010,
        340,
        "arxiv_abs__gutter",
        "empty_boundary",
        "",
    ),
    (
        "0097fb7657aa9131bc2b52cf2745d0fe",
        900,
        1400,
        "arxiv_abs__tools_blank",
        "empty_clear",
        "",
    ),
    (
        "07cc280d49af04b4003229e21e1fcf44",
        820,
        500,
        "euclid__title",
        "core_inner",
        "Testing for outliers with conformal p-values",
    ),
    (
        "07cc280d49af04b4003229e21e1fcf44",
        700,
        1180,
        "euclid__abstract",
        "multi_frag",
        (
            "This paper studies the construction of p-values for nonparametric outlier "
            "detection, from a multiple-testing perspective. The goal is to test whether "
            "new independent samples belong to the same distribution as a reference data "
            "set or are outliers. We propose a solution based on conformal inference, a "
            "general framework yielding p-values that are marginally valid but mutually "
            "dependent for different test points. We prove these p-values are positively "
            "dependent and enable exact false discovery rate control, although in a "
            "relatively weak marginal sense. We then introduce a new method to compute "
            "p-values that are valid conditionally on the training data and independent of "
            "each other for different test points; this paves the way to stronger type-I "
            "error guarantees. Our results depart from classical conformal inference as we "
            "leverage concentration inequalities rather than combinatorial arguments to "
            "establish our finite-sample guarantees. Further, our techniques also yield a "
            "uniform confidence bound for the false positive rate of any outlier detection "
            "algorithm, as a function of the threshold applied to its raw statistics. "
            "Finally, the relevance of our results is demonstrated by experiments on real "
            "and simulated data."
        ),
    ),
    (
        "07cc280d49af04b4003229e21e1fcf44",
        1610,
        1100,
        "euclid__gutter",
        "empty_boundary",
        "",
    ),
    (
        "07cc280d49af04b4003229e21e1fcf44",
        900,
        200,
        "euclid__cookie_chrome",
        "empty_clear",
        "",
    ),
    (
        "64b7841915a3fc3c754e5f44fd161c41",
        180,
        490,
        "arxiv_home__physics_heading",
        "core_inner",
        "Physics",
    ),
    (
        "64b7841915a3fc3c754e5f44fd161c41",
        900,
        545,
        "arxiv_home__astro_line",
        "multi_frag",
        (
            "Astrophysics (**astro-ph** new, recent, search) Astrophysics of Galaxies; "
            "Cosmology and Nongalactic Astrophysics; Earth and Planetary Astrophysics; "
            "High Energy Astrophysical Phenomena; Instrumentation and Methods for "
            "Astrophysics; Solar and Stellar Astrophysics"
        ),
    ),
    (
        "68a0a45aae3e64dd37a772fed83d90c7",
        520,
        640,
        "euclid_ack__group",
        "semantic_group",
        (
            "## Acknowledgments\n\n"
            "We are grateful to the anonymous referees and Associate Editor for their "
            "helpful comments and suggestions."
        ),
    ),
    (
        "68a0a45aae3e64dd37a772fed83d90c7",
        1640,
        720,
        "euclid_ack__gutter",
        "empty_boundary",
        "",
    ),
    (
        "68a0a45aae3e64dd37a772fed83d90c7",
        2100,
        640,
        "euclid_ack__ims_logo",
        "empty_clear",
        "",
    ),
    (
        "7c75e118ee192087b0b2581c60ffaad3",
        720,
        560,
        "wiki__lead",
        "multi_frag",
        (
            "Wikipedi se a free-content online encyclopedia wey community of volunteers, "
            "dem know as Wikipedians dey wrep den dey maintain am, thru open collaboration "
            "den de wiki software MediaWiki. Wikipedia be de largest den most-read "
            "reference work insyd history. wey ebe consistently ranked among de ten most "
            "visited websites; as of August 2024, na dem rank am fourth by Semrush, den "
            "seventh by Similarweb. Ebe founded by Jimmy Wales den Larry Sanger on January "
            "15, 2001, na de Wikimedia Foundation an American nonprofit organization dey "
            "host Wikipedia since 2003, funded mainly by donations from readers."
        ),
    ),
    (
        "7c75e118ee192087b0b2581c60ffaad3",
        1540,
        560,
        "wiki__gutter",
        "empty_boundary",
        "",
    ),
    (
        "7c75e118ee192087b0b2581c60ffaad3",
        1700,
        620,
        "wiki__infobox_image",
        "empty_clear",
        "",
    ),
]


def _emit(
    *,
    img: Image.Image,
    x: int,
    y: int,
    sid: str,
    kind: str,
    target: str,
    page_id: str,
    marked_dir: Path,
    source: str,
    crop_top: int = 0,
) -> PointSample:
    w, h = img.size
    marked = draw_crosshair(img, x, y, spec_for_image(w, h))
    jpg = marked_dir / f"{sid}.jpg"
    marked.save(jpg, format="JPEG", quality=92)
    nx, ny = pixel_to_norm(x, y, w, h)
    neg = target.strip() == ""
    return PointSample(
        sample_id=sid,
        image_path=str(jpg.resolve()),
        task="POINT",
        target=target,
        meta={
            "pool_id": kind,
            "template_stem": "real_world",
            "prompt_key": "a2_v2",
            "marker": CURRENT_MARKER_TAG,
            "page_id": page_id,
            "point": [x, y],
            "point_norm": [nx, ny],
            "image_w": w,
            "image_h": h,
            "region": "neg" if neg else "inner_area",
            "is_negative": neg,
            "source": source,
            "browser_chrome_top_px": crop_top,
        },
    )


def main() -> None:
    OUT_MARKED.mkdir(parents=True, exist_ok=True)
    OUT_NOCHROME.mkdir(parents=True, exist_ok=True)
    OUT_MARKED_NOCHROME.mkdir(parents=True, exist_ok=True)

    crop_top = BROWSER_CHROME_TOP_PX
    samples_full: list[PointSample] = []
    samples_nc: list[PointSample] = []
    cropped_cache: dict[str, Image.Image] = {}

    for stem, x, y, sid, kind, target in CASES:
        src = SRC / f"{stem}.png"
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        full = Image.open(src).convert("RGB")
        samples_full.append(
            _emit(
                img=full,
                x=x,
                y=y,
                sid=sid,
                kind=kind,
                target=target,
                page_id=stem,
                marked_dir=OUT_MARKED,
                source="real_world_sample",
                crop_top=0,
            )
        )
        print(f"  [full] {sid:28s} {kind:16s} ({x},{y}) {'∅' if not target.strip() else 'text'}", flush=True)

        if stem not in cropped_cache:
            cropped = full.crop((0, crop_top, full.size[0], full.size[1]))
            cropped_cache[stem] = cropped
            cropped.save(OUT_NOCHROME / f"{stem}.png")
        cropped = cropped_cache[stem]
        y_nc = y - crop_top
        if y_nc < 0 or y_nc >= cropped.size[1]:
            print(f"  [skip nochrome] {sid}: y={y} falls inside cropped chrome", flush=True)
            continue
        samples_nc.append(
            _emit(
                img=cropped,
                x=x,
                y=y_nc,
                sid=f"{sid}__nochrome",
                kind=kind,
                target=target,
                page_id=f"{stem}__nochrome",
                marked_dir=OUT_MARKED_NOCHROME,
                source="real_world_sample_nochrome",
                crop_top=crop_top,
            )
        )
        print(
            f"  [noc ] {sid:28s} {kind:16s} ({x},{y_nc}) {'∅' if not target.strip() else 'text'}",
            flush=True,
        )

    n_full = write_jsonl(OUT_JSONL, samples_full)
    n_nc = write_jsonl(OUT_JSONL_NOCHROME, samples_nc)
    print(f"wrote {n_full} → {OUT_JSONL}", flush=True)
    print(f"wrote {n_nc} → {OUT_JSONL_NOCHROME} (crop_top={crop_top})", flush=True)


if __name__ == "__main__":
    main()

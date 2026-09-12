"""CLI entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from point_ocr.build_point import BlockAnno, build_point_samples_for_page, load_blocks_json
from point_ocr.dataset_format import write_jsonl
from point_ocr.marker import MARKER_SPEC, draw_crosshair
from point_ocr.sample_points import BBox


def marker_demo() -> None:
    parser = argparse.ArgumentParser(description="Draw MARKER_SPEC crosshair on an image")
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    args = parser.parse_args()
    img = Image.open(args.image)
    out = draw_crosshair(img, args.x, args.y, MARKER_SPEC)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(json.dumps({"out": str(args.out), "spec": MARKER_SPEC.to_dict()}, indent=2))


def build_synth() -> None:
    parser = argparse.ArgumentParser(description="Render HTML templates → POINT samples")
    parser.add_argument("--templates", type=Path, required=True, help="Dir of *.html templates")
    parser.add_argument("--out", type=Path, required=True, help="Output root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r-min", type=int, default=2)
    parser.add_argument("--r-max", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=4)
    args = parser.parse_args()

    from point_ocr.synth.render import render_html_file_sync

    render_dir = args.out / "renders"
    images_dir = args.out / "marked"
    samples_path = args.out / "point_sharegpt.jsonl"
    render_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    html_files = sorted(args.templates.glob("*.html"))
    for i, html in enumerate(html_files):
        meta = render_html_file_sync(html, render_dir, page_id=html.stem)
        img = Image.open(meta["image_path"])
        blocks_raw = load_blocks_json(Path(meta["blocks_path"]))
        page_samples = build_point_samples_for_page(
            img,
            blocks_raw,
            page_id=html.stem,
            out_image_dir=images_dir,
            r_min=args.r_min,
            r_max=args.r_max,
            n_negatives=args.negatives,
            seed=args.seed + i,
        )
        all_samples.extend(page_samples)
        print(f"[ok] {html.name}: {len(blocks_raw)} blocks → {len(page_samples)} samples")

    n = write_jsonl(samples_path, all_samples, fmt="sharegpt")
    print(f"Wrote {n} samples → {samples_path}")


def run_eval() -> None:
    parser = argparse.ArgumentParser(description="Score predictions JSONL against targets")
    parser.add_argument("--pred", type=Path, required=True, help="JSONL: sample_id,prediction,target,is_negative?")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--hit-threshold", type=float, default=0.85)
    args = parser.parse_args()

    from point_ocr.dataset_format import load_jsonl
    from point_ocr.infer import strip_format_leak
    from point_ocr.metrics import EvalExample, evaluate_examples

    rows = load_jsonl(args.pred)
    examples = []
    for i, r in enumerate(rows):
        raw = str(r.get("prediction_raw", r.get("prediction", r.get("pred", ""))))
        scored = str(r.get("prediction", r.get("pred", "")))
        # Prefer explicit cleaned field; otherwise strip leaks from whatever we have.
        if "prediction_raw" in r:
            cleaned = scored
            raw_for_leak = raw
        else:
            out = strip_format_leak(scored)
            cleaned = out.cleaned
            raw_for_leak = out.raw
        examples.append(
            EvalExample(
                sample_id=str(r.get("sample_id", i)),
                prediction=cleaned,
                target=str(r.get("target", r.get("label", ""))),
                is_negative=bool(r.get("is_negative", False)),
                raw_prediction=raw_for_leak,
            )
        )
    report = evaluate_examples(examples, hit_threshold=args.hit_threshold)
    text = json.dumps(report.to_dict(), indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


# silence unused import warning for BlockAnno/BBox in type-checkers
_ = (BlockAnno, BBox)

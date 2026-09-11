#!/usr/bin/env python3
"""Build real-arm POINT samples from page images + layout + OvisOCR2 crop labels.

Pipeline:
  1) Layout model → block bboxes (plug in your detector; stub available)
  2) OvisOCR2 OCR on each crop → markdown GT
  3) Multi-point crosshair samples + negatives

Privacy: keep raw screenshots private; do not publish without rights clearance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from point_ocr.build_point import BlockAnno, build_point_samples_for_page
from point_ocr.dataset_format import write_jsonl
from point_ocr.filter_qa import filter_block_label
from point_ocr.noise import apply_screen_noise
from point_ocr.real import StubLayoutDetector, label_page_with_crops
from point_ocr.sample_points import BBox


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True, help="Dir of page screenshots (png/jpg)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "real")
    ap.add_argument("--use-stub-layout", action="store_true", help="Dev only: fake full-page box")
    ap.add_argument("--use-ovis-ocr", action="store_true", help="Require GPU vLLM OvisOCR2 crop OCR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--r-min", type=int, default=2)
    ap.add_argument("--r-max", type=int, default=5)
    ap.add_argument("--negatives", type=int, default=4)
    args = ap.parse_args()

    if not args.use_stub_layout:
        raise SystemExit(
            "Provide a LayoutDetector implementation and wire it here, "
            "or pass --use-stub-layout for smoke tests only."
        )

    layout = StubLayoutDetector()
    if args.use_ovis_ocr:
        from point_ocr.real import OvisOCR2CropLabeler

        ocr = OvisOCR2CropLabeler()
    else:
        class EchoOCR:
            def ocr_crop(self, image: Image.Image) -> str:
                return ""  # forces filter drop unless you replace

        ocr = EchoOCR()
        print("[warn] without --use-ovis-ocr, labels are empty and will be filtered out")

    images_dir = args.out / "marked"
    ann_dir = args.out / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    all_samples = []

    paths = sorted(
        list(args.images.glob("*.png"))
        + list(args.images.glob("*.jpg"))
        + list(args.images.glob("*.jpeg"))
    )
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        dets = layout.detect(img)
        blocks_raw = label_page_with_crops(img, dets, ocr)
        (ann_dir / f"{path.stem}.blocks.json").write_text(
            json.dumps(blocks_raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if args.noise:
            img = apply_screen_noise(img)

        blocks: list[BlockAnno] = []
        for row in blocks_raw:
            fr = filter_block_label(row["markdown"], tag=row.get("label"))
            if not fr.keep:
                continue
            x0, y0, x1, y1 = row["bbox"]
            blocks.append(
                BlockAnno(
                    block_id=str(row["id"]),
                    bbox=BBox(float(x0), float(y0), float(x1), float(y1)),
                    markdown=row["markdown"],
                )
            )

        page_samples = build_point_samples_for_page(
            img,
            blocks,
            page_id=path.stem,
            out_image_dir=images_dir,
            r_min=args.r_min,
            r_max=args.r_max,
            n_negatives=args.negatives,
            seed=args.seed + i,
        )
        all_samples.extend(page_samples)
        print(f"[ok] {path.name}: blocks={len(blocks)} samples={len(page_samples)}")

    out_jsonl = args.out / "point_sharegpt.jsonl"
    n = write_jsonl(out_jsonl, all_samples)
    print(f"Wrote {n} samples → {out_jsonl}")


if __name__ == "__main__":
    main()

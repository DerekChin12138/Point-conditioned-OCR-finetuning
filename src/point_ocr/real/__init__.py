"""Real-document arm: layout bboxes + OvisOCR2 crop labeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from point_ocr.sample_points import BBox


class LayoutDetector(Protocol):
    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        """Return list of {id, bbox:[x0,y0,x1,y1], score?, label?}."""
        ...


class CropOCR(Protocol):
    def ocr_crop(self, image: Image.Image) -> str:
        """OCR a cropped block image → Markdown/text label."""
        ...


class StubLayoutDetector:
    """Placeholder layout detector for wiring tests.

    Replace with DocLayout-YOLO / PP-DocLayout / etc. in production.
    """

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        w, h = image.size
        # Single fake full-content region (not for real training)
        return [{"id": "stub0", "bbox": [w * 0.1, h * 0.1, w * 0.9, h * 0.9], "label": "text"}]


class OvisOCR2CropLabeler:
    """Use OvisOCR2 (vLLM) on block crops as content GT (preferred over Paddle/MinerU).

    Requires CUDA + vllm on Linux/Windows training machine.
    """

    def __init__(self, model_name: str = "ATH-MaaS/OvisOCR2"):
        self.model_name = model_name
        self._parser = None

    def _ensure(self) -> None:
        if self._parser is not None:
            return
        # Lazy import — only on GPU hosts
        from point_ocr.real.ovis_crop_client import OvisCropClient

        self._parser = OvisCropClient(self.model_name)

    def ocr_crop(self, image: Image.Image) -> str:
        self._ensure()
        assert self._parser is not None
        return self._parser.parse_crop(image)


def label_page_with_crops(
    image: Image.Image,
    detections: list[dict[str, Any]],
    ocr: CropOCR,
    *,
    pad: int = 2,
) -> list[dict[str, Any]]:
    """For each layout box, crop → OCR → markdown annotation row."""
    w, h = image.size
    out: list[dict[str, Any]] = []
    for i, det in enumerate(detections):
        x0, y0, x1, y1 = det["bbox"]
        x0i = max(0, int(x0) - pad)
        y0i = max(0, int(y0) - pad)
        x1i = min(w, int(x1) + pad)
        y1i = min(h, int(y1) + pad)
        if x1i <= x0i or y1i <= y0i:
            continue
        crop = image.crop((x0i, y0i, x1i, y1i))
        text = ocr.ocr_crop(crop).strip()
        bid = str(det.get("id", f"b{i}"))
        out.append(
            {
                "id": bid,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "markdown": text,
                "label": det.get("label"),
                "score": det.get("score"),
            }
        )
    return out


def save_blocks(path: Path, blocks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")


def blocks_to_bboxes(blocks: list[dict[str, Any]]) -> list[BBox]:
    return [BBox(*map(float, b["bbox"])) for b in blocks]

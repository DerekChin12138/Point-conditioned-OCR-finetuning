"""Offline POINT sample builder (no Playwright)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from point_ocr.build_point import BlockAnno, build_point_samples_for_page
from point_ocr.html_to_md import html_fragment_to_markdown
from point_ocr.sample_points import BBox


def test_html_to_md_paragraph():
    md = html_fragment_to_markdown('<p data-block-id="p1">Hello <b>world</b>.</p>')
    assert "Hello" in md and "world" in md


def test_build_point_samples(tmp_path: Path):
    img = Image.new("RGB", (640, 480), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 60, 400, 140], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((100, 80), "Title block", fill=(0, 0, 0))
    draw.rectangle([80, 180, 520, 320], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((100, 220), "Body paragraph text.", fill=(0, 0, 0))

    blocks = [
        BlockAnno("t", BBox(80, 60, 400, 140), "# Title block"),
        BlockAnno("b", BBox(80, 180, 520, 320), "Body paragraph text."),
    ]
    samples = build_point_samples_for_page(
        img,
        blocks,
        page_id="demo",
        out_image_dir=tmp_path / "marked",
        r_min=2,
        r_max=2,
        n_negatives=2,
        seed=0,
    )
    # 2 blocks * 2 points + 2 negatives
    assert len(samples) == 6
    assert any(s.target == "" for s in samples)
    assert all(Path(s.image_path).exists() for s in samples)

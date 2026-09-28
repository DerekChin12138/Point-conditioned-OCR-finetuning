"""Chrome wrap is a third layer; none is identity."""

from __future__ import annotations

import random

from point_ocr.chrome.wrap import ChromeSpec, sample_chrome_spec, wrap_page_html


_MINI = """<!DOCTYPE html><html><head><title>t</title></head>
<body><p data-block-id="p1">Hello body paragraph for tests.</p></body></html>"""


def test_none_is_identity():
    spec = ChromeSpec(family="none", skin="none")
    assert wrap_page_html(_MINI, spec) == _MINI


def test_browser_wrap_adds_bands_not_block_ids():
    rng = random.Random(0)
    spec = sample_chrome_spec(rng, family_weights={"none": 0, "browser": 1, "office": 0})
    assert spec.family == "browser"
    out = wrap_page_html(_MINI, spec)
    assert "data-chrome-band" in out
    assert 'data-block-id="p1"' in out
    assert "poc-chrome" in out
    # chrome copy must not become labeled blocks
    assert out.count("data-block-id") == 1


def test_wrap_does_not_paint_document_with_chrome_type():
    src = (
        "<!DOCTYPE html><html><head><style>"
        "body { background:#e8e6e1; color:#1a1a1a; font-size:16px; }"
        "p { font-size:15px; }</style></head>"
        "<body><p data-block-id='p1'>Hello body paragraph for tests.</p></body></html>"
    )
    spec = sample_chrome_spec(
        random.Random(0), family_weights={"none": 0, "browser": 1, "office": 0}
    )
    spec.skin = "chrome_win_dark"
    out = wrap_page_html(src, spec)
    assert ".poc-client{background:#e8e6e1 !important;color:#1a1a1a !important;}" in out
    chrome_root = [ln for ln in out.splitlines() if ".poc-chrome {" in ln]
    assert chrome_root
    assert "font-size:12px" not in chrome_root[0]


def test_wrap_copies_first_body_rule_for_dark_templates():
    """GitHub-like templates put body { bg; color } first — no preceding }."""
    src = (
        "<!DOCTYPE html><html><head><style>"
        "body { margin:0; background:#0d1117; color:#e6edf3; }"
        "p, li { color:#c9d1d9; }"
        "</style></head><body><p data-block-id='p1'>readme</p></body></html>"
    )
    for skin, family in (
        ("safari_mac", "browser"),
        ("word_win", "office"),
        ("wps_win", "office"),
    ):
        spec = ChromeSpec(family=family, skin=skin, show_status=True)
        out = wrap_page_html(src, spec)
        assert ".poc-client{background:#0d1117 !important;color:#e6edf3 !important;}" in out, skin


def test_office_wrap_has_head_and_foot():
    rng = random.Random(1)
    spec = sample_chrome_spec(rng, family_weights={"none": 0, "browser": 0, "office": 1})
    spec.show_status = True
    spec.show_sidebar = True
    out = wrap_page_html(_MINI, spec)
    assert 'data-chrome-band="head"' in out
    assert 'data-chrome-band="foot"' in out
    assert 'data-chrome-band="side"' in out

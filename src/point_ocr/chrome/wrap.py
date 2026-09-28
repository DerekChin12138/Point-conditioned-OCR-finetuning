"""Window-chrome as a third data-engine layer (not a replacement template set).

Layer 1: content diversity (filled HTML).
Layer 2: hard-case pools (core_inner / empty_* / …).
Layer 3: optional OS/app chrome wrapped around the page at render time.

Chrome elements must not carry ``data-block-id`` — they are scenery / empty-label.
"""

from __future__ import annotations

import html
import random
import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup

CHROME_FAMILIES = ("none", "browser", "office")
BROWSER_SKINS = ("chrome_win_light", "chrome_win_dark", "safari_mac", "firefox_linux")
OFFICE_SKINS = ("word_win", "wps_win")

# When chrome is applied, keep a large none share so pools are not all shelled.
DEFAULT_FAMILY_WEIGHTS: dict[str, float] = {
    "none": 0.45,
    "browser": 0.35,
    "office": 0.20,
}

# B1: 60% none / 40% chrome; among chrome pages keep browser:office = 35:20.
B1_FAMILY_WEIGHTS: dict[str, float] = {
    "none": 0.60,
    "browser": 0.40 * 35.0 / 55.0,
    "office": 0.40 * 20.0 / 55.0,
}


@dataclass
class ChromeSpec:
    family: str = "none"
    skin: str = "none"
    theme: str = "light"
    show_bookmarks: bool = True
    show_sidebar: bool = False
    show_status: bool = True
    content: dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> dict[str, Any]:
        return {
            "chrome_family": self.family,
            "chrome_skin": self.skin,
            "chrome_theme": self.theme,
            "chrome_bookmarks": self.show_bookmarks,
            "chrome_sidebar": self.show_sidebar,
            "chrome_status": self.show_status,
        }


def _e(s: Any) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def _pick(items: list, rng: random.Random, default: Any) -> Any:
    if not items:
        return default
    return rng.choice(items)


def sample_chrome_spec(
    rng: random.Random,
    pool: dict[str, Any] | None = None,
    *,
    family_weights: dict[str, float] | None = None,
) -> ChromeSpec:
    """Sample layer-3 chrome. ``none`` is a first-class outcome."""
    pool = pool or {}
    weights = family_weights or DEFAULT_FAMILY_WEIGHTS
    families = list(weights.keys())
    wts = [max(0.0, float(weights[f])) for f in families]
    family = rng.choices(families, weights=wts, k=1)[0]
    if family == "none":
        return ChromeSpec(family="none", skin="none")
    if family == "browser":
        skin = rng.choice(list(BROWSER_SKINS))
        theme = "dark" if "dark" in skin else "light"
        scene = _pick(list(pool.get("browser_scenes") or []), rng, _fallback_browser(rng))
        if not isinstance(scene, dict):
            scene = _fallback_browser(rng)
        return ChromeSpec(
            family="browser",
            skin=skin,
            theme=theme,
            show_bookmarks=rng.random() < 0.7,
            show_sidebar=rng.random() < 0.22,
            show_status=rng.random() < 0.35,
            content=dict(scene),
        )
    skin = rng.choice(list(OFFICE_SKINS))
    scene = _pick(list(pool.get("office_scenes") or []), rng, _fallback_office(rng, skin))
    if not isinstance(scene, dict):
        scene = _fallback_office(rng, skin)
    return ChromeSpec(
        family="office",
        skin=skin,
        theme="light",
        show_bookmarks=False,
        show_sidebar=rng.random() < 0.45,
        show_status=True,
        content=dict(scene),
    )


def _fallback_browser(rng: random.Random) -> dict[str, Any]:
    tabs = rng.choice(
        [
            ["arxiv.org", "A Ranking Approach for Measuring Calibration", "Gmail"],
            ["知乎", "Wikipedia", "DeepL", "localhost:7865"],
            ["Google Scholar", "GitHub", "Stack Overflow"],
            ["Bing", "微软文档", "Outlook"],
        ]
    )
    urls = [
        "https://arxiv.org/abs/2609.13100",
        "https://en.wikipedia.org/wiki/Optical_character_recognition",
        "https://github.com/search?q=ovisocr2",
        "https://www.zhihu.com/question/123456",
        "https://scholar.google.com/scholar?q=calibration+error",
    ]
    return {
        "tabs": tabs,
        "active": min(1, len(tabs) - 1),
        "url": rng.choice(urls),
        "bookmarks": rng.choice(
            [["Gmail", "Drive", "翻译", "Calendar"], ["知乎", "Bilibili", "微信读书"], ["Docs", "Sheets", "Meet"]]
        ),
        "profile": rng.choice(["林晓", "Alex Chen", "王倩", "M. Patel"]),
    }


def _fallback_office(rng: random.Random, skin: str) -> dict[str, Any]:
    names = [
        "开题报告_v3.docx",
        "Q3_Board_Pack.docx",
        "实验记录-09-16.docx",
        "Meeting notes 16 Sep.docx",
        "合同审核稿（最终）.docx",
    ]
    return {
        "app": "WPS 文字" if skin == "wps_win" else "Word",
        "filename": rng.choice(names),
        "user": rng.choice(["王倩", "Derek Liu", "陈可", "Sofia R."]),
        "page": rng.choice(["1 / 12", "3 / 18", "7 / 7", "2 / 41"]),
        "words": rng.choice(["1284", "4520", "891", "12003"]),
        "zoom": rng.choice(["100%", "110%", "90%", "125%"]),
        "lang": "zh" if rng.random() < 0.55 else "en",
    }


def wrap_page_html(page_html: str, spec: ChromeSpec) -> str:
    """Inject window chrome around existing body content. Identity if family=none."""
    if spec.family == "none":
        return page_html
    soup = BeautifulSoup(page_html, "lxml")
    if soup.html is None:
        soup = BeautifulSoup("<html><head></head><body></body></html>" + page_html, "lxml")
    if soup.head is None:
        soup.html.insert(0, soup.new_tag("head"))
    if soup.body is None:
        soup.html.append(soup.new_tag("body"))

    client = soup.new_tag("div", attrs={"class": "poc-client"})
    for child in list(soup.body.contents):
        client.append(child.extract())

    css = soup.new_tag("style")
    css.string = (
        _CHROME_BASE_CSS
        + "\n"
        + _skin_css(spec)
        + "\n"
        + _body_canvas_css(page_html)
    )
    soup.head.append(css)

    root = soup.new_tag(
        "div",
        attrs={
            "class": f"poc-chrome poc-{spec.skin}",
            "data-chrome-family": spec.family,
            "data-chrome-skin": spec.skin,
        },
    )
    head = BeautifulSoup(_head_html(spec), "html.parser")
    for el in list(head.contents):
        root.append(el)
    mid = soup.new_tag("div", attrs={"class": "poc-mid"})
    if spec.show_sidebar:
        side = BeautifulSoup(_side_html(spec), "html.parser")
        for el in list(side.contents):
            mid.append(el)
    mid.append(client)
    root.append(mid)
    if spec.show_status:
        foot = BeautifulSoup(_foot_html(spec), "html.parser")
        for el in list(foot.contents):
            root.append(el)
    soup.body.append(root)
    soup.body["class"] = (soup.body.get("class") or []) + ["poc-has-chrome"]
    return str(soup)


# First ``body {`` in a <style> block has no preceding ``}``; ``.body`` / ``#body`` must not match.
_BODY_RULE = re.compile(r"(?<![.#\w-])body\s*\{([^}]+)\}", re.I | re.S)


def _body_canvas_css(page_html: str) -> str:
    """Copy original body background/color onto the document pane.

    Wrapping moves body children into ``.poc-client``. Body ``background`` would
    otherwise sit behind the window chrome. A white client plus a dark-theme
    ``color`` (or ``p { color:#c9d1d9 }``) is light-gray type on white.
    """
    bg: str | None = None
    fg: str | None = None
    for m in _BODY_RULE.finditer(page_html):
        block = m.group(1)
        bm = re.search(r"background(?:-color)?\s*:\s*([^;]+)", block, re.I)
        cm = re.search(r"(?<![a-z-])color\s*:\s*([^;]+)", block, re.I)
        if bm:
            val = bm.group(1).strip()
            if val.lower() not in ("none", "inherit", "transparent"):
                bg = val
        if cm:
            fg = cm.group(1).strip()
    if not bg and not fg:
        return ""
    parts = []
    if bg:
        parts.append(f"background:{bg} !important")
    if fg:
        parts.append(f"color:{fg} !important")
    return ".poc-client{" + ";".join(parts) + ";}"


_CHROME_BASE_CSS = """
html, body.poc-has-chrome { margin:0 !important; height:100% !important; overflow:hidden !important; }
body.poc-has-chrome { min-height:100vh !important; }
.poc-chrome { display:flex; flex-direction:column; height:100vh; overflow:hidden; }
.poc-caption, .poc-tabs, .poc-toolbar, .poc-bookmarks, .poc-ribbon, .poc-status, .poc-side,
.poc-cap-btns, .poc-tab, .poc-bm, .poc-navbtn, .poc-omnibox, .poc-rtab, .poc-grp {
  font-family:"Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  font-size:12px; user-select:none; box-sizing:border-box;
}
.poc-mid { display:flex; flex:1 1 auto; min-height:0; min-width:0; }
.poc-client { flex:1 1 auto; min-width:0; min-height:0; overflow:auto; background:#fff; }
.poc-chrome [data-chrome-band] { flex:0 0 auto; }
.poc-dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
.poc-cap-btns { display:flex; margin-left:auto; height:100%; }
.poc-cap-btns span { width:46px; display:flex; align-items:center; justify-content:center;
  font-size:11px; color:#5f6368; }
.poc-omnibox { flex:1; height:28px; border-radius:14px; border:1px solid #dadce0;
  background:#fff; display:flex; align-items:center; padding:0 12px; gap:8px;
  color:#202124; min-width:0; margin:0 12px; }
.poc-omnibox .url { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.poc-tab { padding:6px 14px; border-radius:8px 8px 0 0; max-width:180px;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.poc-tab.active { background:#fff; }
.poc-bm { padding:4px 8px; color:#3c4043; }
.poc-navbtn { width:28px; height:28px; border-radius:50%; display:flex;
  align-items:center; justify-content:center; color:#5f6368; }
"""


def _skin_css(spec: ChromeSpec) -> str:
    if spec.skin == "chrome_win_dark":
        return """
.poc-chrome_win_dark { background:#202124; }
.poc-chrome_win_dark [data-chrome-band] { color:#e8eaed; }
.poc-chrome_win_dark .poc-caption { height:32px; background:#202124; display:flex; align-items:center; padding:0 8px; }
.poc-chrome_win_dark .poc-tabs { height:34px; background:#202124; display:flex; align-items:flex-end; padding:0 8px; gap:2px; }
.poc-chrome_win_dark .poc-tab { background:#2b2c2f; color:#9aa0a6; }
.poc-chrome_win_dark .poc-tab.active { background:#35363a; color:#e8eaed; }
.poc-chrome_win_dark .poc-toolbar { height:40px; background:#35363a; display:flex; align-items:center; padding:0 8px; }
.poc-chrome_win_dark .poc-omnibox { background:#202124; border-color:#5f6368; color:#e8eaed; }
.poc-chrome_win_dark .poc-bookmarks { height:28px; background:#35363a; display:flex; align-items:center; padding:0 8px; }
.poc-chrome_win_dark .poc-bm { color:#9aa0a6; }
.poc-chrome_win_dark .poc-status { height:22px; background:#202124; color:#9aa0a6; display:flex; align-items:center; padding:0 10px; border-top:1px solid #3c4043; }
.poc-chrome_win_dark .poc-side { width:52px; background:#202124; border-right:1px solid #3c4043; }
.poc-chrome_win_dark .poc-cap-btns span { color:#e8eaed; }
"""
    if spec.skin == "safari_mac":
        return """
.poc-safari_mac { background:#ececec; }
.poc-safari_mac [data-chrome-band] { color:#1d1d1f; }
.poc-safari_mac .poc-caption { height:52px; background:linear-gradient(#f6f6f6,#e9e9e9);
  display:flex; align-items:center; padding:0 12px; gap:8px; border-bottom:1px solid #d0d0d0; }
.poc-safari_mac .poc-toolbar { display:none; }
.poc-safari_mac .poc-tabs { display:none; }
.poc-safari_mac .poc-omnibox { height:26px; border-radius:8px; background:#fff; border:1px solid #c7c7c7;
  box-shadow: inset 0 1px 0 #fff; max-width:560px; margin:0 auto; }
.poc-safari_mac .poc-bookmarks { height:28px; background:#f4f4f4; border-bottom:1px solid #d8d8d8;
  display:flex; align-items:center; padding:0 16px; }
.poc-safari_mac .poc-status { height:0; overflow:hidden; padding:0; border:0; }
.poc-safari_mac .poc-side { width:180px; background:#f5f5f7; border-right:1px solid #d2d2d7; padding:10px 8px; color:#6e6e73; }
"""
    if spec.skin == "firefox_linux":
        return """
.poc-firefox_linux { background:#38383d; }
.poc-firefox_linux [data-chrome-band] { color:#f9f9fa; }
.poc-firefox_linux .poc-caption { height:36px; background:#38383d; display:flex; align-items:center; padding:0 6px; }
.poc-firefox_linux .poc-tabs { height:32px; background:#2b2a33; display:flex; align-items:flex-end; padding:0 6px; gap:1px; }
.poc-firefox_linux .poc-tab { background:#1c1b22; color:#cfcfd8; border-radius:4px 4px 0 0; }
.poc-firefox_linux .poc-tab.active { background:#42414d; color:#fff; }
.poc-firefox_linux .poc-toolbar { height:38px; background:#42414d; display:flex; align-items:center; padding:0 8px; }
.poc-firefox_linux .poc-omnibox { background:#1c1b22; border-color:#5b5b66; color:#f9f9fa; border-radius:4px; }
.poc-firefox_linux .poc-bookmarks { height:26px; background:#2b2a33; display:flex; align-items:center; padding:0 8px; }
.poc-firefox_linux .poc-bm { color:#cfcfd8; }
.poc-firefox_linux .poc-status { height:22px; background:#2b2a33; color:#cfcfd8; display:flex; align-items:center; padding:0 8px; }
.poc-firefox_linux .poc-side { width:48px; background:#2b2a33; }
"""
    if spec.skin == "word_win":
        return """
.poc-word_win { background:#f3f3f3; }
.poc-word_win [data-chrome-band] { color:#242424; }
.poc-word_win .poc-caption { height:30px; background:#f3f3f3; display:flex; align-items:center; padding:0 8px; gap:8px; }
.poc-word_win .poc-file { background:#0f703b; color:#fff; padding:4px 14px; font-size:12px; }
.poc-word_win .poc-ribbon { background:#fff; border-bottom:1px solid #e1e1e1; }
.poc-word_win .poc-rtabs { height:30px; display:flex; align-items:flex-end; padding:0 8px; gap:2px; }
.poc-word_win .poc-rtab { padding:6px 12px; }
.poc-word_win .poc-rtab.active { border-bottom:2px solid #0f703b; font-weight:600; }
.poc-word_win .poc-rtools { height:72px; display:flex; gap:10px; padding:6px 10px; }
.poc-word_win .poc-grp { border-right:1px solid #e1e1e1; padding:0 10px; min-width:72px; }
.poc-word_win .poc-grp b { display:block; font-size:10px; color:#616161; text-align:center; margin-top:4px; font-weight:400; }
.poc-word_win .poc-status { height:22px; background:#0f703b; color:#fff; display:flex; align-items:center; padding:0 10px; gap:16px; }
.poc-word_win .poc-side { width:200px; background:#fafafa; border-right:1px solid #e1e1e1; padding:8px; font-size:12px; color:#616161; }
"""
    if spec.skin == "wps_win":
        return """
.poc-wps_win { background:#f5f5f5; }
.poc-wps_win [data-chrome-band] { color:#333; }
.poc-wps_win .poc-caption { height:32px; background:#d83313; color:#fff; display:flex; align-items:center; padding:0 8px; gap:8px; }
.poc-wps_win .poc-cap-btns span { color:#fff; }
.poc-wps_win .poc-ribbon { background:#fff; border-bottom:1px solid #eee; }
.poc-wps_win .poc-rtabs { height:30px; display:flex; padding:0 8px; }
.poc-wps_win .poc-rtab { padding:6px 12px; }
.poc-wps_win .poc-rtab.active { color:#d83313; border-bottom:2px solid #d83313; font-weight:600; }
.poc-wps_win .poc-rtools { height:68px; display:flex; gap:8px; padding:6px 10px; }
.poc-wps_win .poc-grp { border-right:1px solid #eee; padding:0 10px; }
.poc-wps_win .poc-grp b { display:block; font-size:10px; color:#888; text-align:center; margin-top:4px; font-weight:400; }
.poc-wps_win .poc-status { height:22px; background:#f0f0f0; color:#666; display:flex; align-items:center; padding:0 10px; gap:14px; border-top:1px solid #e5e5e5; }
.poc-wps_win .poc-side { width:188px; background:#fafafa; border-right:1px solid #eee; padding:8px; color:#666; }
"""
    # chrome_win_light default
    return """
.poc-chrome_win_light { background:#dee1e6; }
.poc-chrome_win_light [data-chrome-band] { color:#202124; }
.poc-chrome_win_light .poc-caption { height:32px; background:#dee1e6; display:flex; align-items:center; padding:0 6px; }
.poc-chrome_win_light .poc-tabs { height:34px; background:#dee1e6; display:flex; align-items:flex-end; padding:0 8px; gap:2px; }
.poc-chrome_win_light .poc-tab { background:#cfd1d4; color:#3c4043; }
.poc-chrome_win_light .poc-tab.active { background:#fff; color:#202124; }
.poc-chrome_win_light .poc-toolbar { height:40px; background:#fff; display:flex; align-items:center; padding:0 8px; border-bottom:1px solid #e8eaed; }
.poc-chrome_win_light .poc-bookmarks { height:28px; background:#fff; display:flex; align-items:center; padding:0 8px; border-bottom:1px solid #e8eaed; }
.poc-chrome_win_light .poc-status { height:22px; background:#fff; color:#5f6368; display:flex; align-items:center; padding:0 10px; border-top:1px solid #e8eaed; }
.poc-chrome_win_light .poc-side { width:52px; background:#fff; border-right:1px solid #e8eaed; }
"""


def _head_html(spec: ChromeSpec) -> str:
    if spec.family == "office":
        return _office_head(spec)
    return _browser_head(spec)


def _browser_head(spec: ChromeSpec) -> str:
    c = spec.content
    tabs = list(c.get("tabs") or ["New Tab"])
    active = int(c.get("active") or 0) % max(len(tabs), 1)
    url = str(c.get("url") or "https://example.com")
    profile = str(c.get("profile") or "")
    tab_html = []
    for i, t in enumerate(tabs[:5]):
        cls = "poc-tab active" if i == active else "poc-tab"
        tab_html.append(f'<div class="{cls}">{_e(t)}</div>')
    bm = ""
    if spec.show_bookmarks:
        marks = "".join(f'<span class="poc-bm">{_e(x)}</span>' for x in (c.get("bookmarks") or [])[:6])
        bm = f'<div class="poc-bookmarks" data-chrome-band="head">{marks}</div>'
    if spec.skin == "safari_mac":
        lights = (
            '<span class="poc-dot" style="background:#ff5f57"></span>'
            '<span class="poc-dot" style="background:#febc2e"></span>'
            '<span class="poc-dot" style="background:#28c840"></span>'
        )
        return (
            f'<div class="poc-caption" data-chrome-band="head">{lights}'
            f'<div class="poc-omnibox"><span class="url">{_e(url)}</span></div></div>{bm}'
        )
    cap_btns = '<div class="poc-cap-btns"><span>—</span><span>□</span><span>×</span></div>'
    title = tabs[active] if tabs else "Browser"
    return (
        f'<div class="poc-caption" data-chrome-band="head"><span>{_e(title)}</span>{cap_btns}</div>'
        f'<div class="poc-tabs" data-chrome-band="head">{"".join(tab_html)}'
        f'<div class="poc-tab">+</div></div>'
        f'<div class="poc-toolbar" data-chrome-band="head">'
        f'<span class="poc-navbtn">←</span><span class="poc-navbtn">→</span>'
        f'<span class="poc-navbtn">↻</span>'
        f'<div class="poc-omnibox"><span>🔒</span><span class="url">{_e(url)}</span></div>'
        f'<span class="poc-navbtn">{_e(profile[:1] or "U")}</span></div>{bm}'
    )


def _office_head(spec: ChromeSpec) -> str:
    c = spec.content
    filename = str(c.get("filename") or "Document.docx")
    app = str(c.get("app") or ("WPS 文字" if spec.skin == "wps_win" else "Word"))
    zh = str(c.get("lang") or "zh") == "zh"
    if spec.skin == "wps_win":
        tabs = ["开始", "插入", "页面", "引用", "审阅", "视图", "工具"] if zh else [
            "Home", "Insert", "Page", "Review", "View"
        ]
        groups = [("粘贴", "剪贴板"), ("字体", "字体"), ("段落", "段落"), ("样式", "样式")] if zh else [
            ("Paste", "Clipboard"), ("Font", "Font"), ("Para", "Paragraph")
        ]
        file_btn = ""
        brand = f'<span style="font-weight:700;margin-right:8px">WPS</span>'
    else:
        tabs = ["开始", "插入", "绘图", "设计", "布局", "引用", "邮件", "审阅", "视图"] if zh else [
            "Home", "Insert", "Draw", "Design", "Layout", "References", "Review", "View"
        ]
        groups = [("粘贴", "剪贴板"), ("B I U", "字体"), ("≡ ☰", "段落"), ("标题", "样式")] if zh else [
            ("Paste", "Clipboard"), ("B I U", "Font"), ("≡", "Paragraph")
        ]
        file_btn = '<span class="poc-file">文件</span>' if zh else '<span class="poc-file">File</span>'
        brand = ""
    tab_html = []
    for i, t in enumerate(tabs):
        cls = "poc-rtab active" if i == 0 else "poc-rtab"
        tab_html.append(f'<span class="{cls}">{_e(t)}</span>')
    grp_html = "".join(
        f'<div class="poc-grp"><div>{_e(a)}</div><b>{_e(b)}</b></div>' for a, b in groups
    )
    cap_btns = '<div class="poc-cap-btns"><span>—</span><span>□</span><span>×</span></div>'
    return (
        f'<div class="poc-caption" data-chrome-band="head">{brand}{file_btn}'
        f'<span style="margin-left:8px">{_e(filename)}  -  {_e(app)}</span>{cap_btns}</div>'
        f'<div class="poc-ribbon" data-chrome-band="head">'
        f'<div class="poc-rtabs">{"".join(tab_html)}</div>'
        f'<div class="poc-rtools">{grp_html}</div></div>'
    )


def _side_html(spec: ChromeSpec) -> str:
    if spec.family == "office":
        items = spec.content.get("outline") or ["标题 1", "引言", "方法", "结果", "参考文献"]
        lis = "".join(f"<div style='padding:4px 6px'>{_e(x)}</div>" for x in items[:8])
        return f'<div class="poc-side" data-chrome-band="side">{lis}</div>'
    icons = "".join(f'<div style="text-align:center;padding:10px 0;color:#9aa0a6">●</div>' for _ in range(4))
    return f'<div class="poc-side" data-chrome-band="side">{icons}</div>'


def _foot_html(spec: ChromeSpec) -> str:
    c = spec.content
    if spec.family == "office":
        bits = [
            str(c.get("page") or "1 / 1"),
            f"{c.get('words') or '0'} 字" if str(c.get("lang") or "zh") == "zh" else f"{c.get('words') or '0'} words",
            str(c.get("zoom") or "100%"),
            str(c.get("user") or ""),
        ]
        inner = "".join(f"<span>{_e(b)}</span>" for b in bits if b)
        return f'<div class="poc-status" data-chrome-band="foot">{inner}</div>'
    return (
        f'<div class="poc-status" data-chrome-band="foot">'
        f'<span>{_e(c.get("url") or "")}</span></div>'
    )

"""Serialize HTML semantic blocks to Markdown targets (source-of-truth for synth)."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify as md


_WS_RE = re.compile(r"[ \t]+\n")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub("\n", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def html_fragment_to_markdown(html: str) -> str:
    """Convert a single block HTML fragment to Markdown GT."""
    html = html.strip()
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")
    # Prefer the first meaningful element if wrapped in <html><body>
    body = soup.body
    root: Tag | NavigableString | None = body if body else soup
    if isinstance(root, Tag):
        # Table cell: keep compact
        if root.name == "td" or (len(root.find_all("td")) == 1 and root.name in {"html", "body", "[document]"}):
            td = root if root.name == "td" else root.find("td")
            if td is not None:
                return normalize_markdown(td.get_text(" ", strip=True))

        # Formula: prefer LaTeX from data-latex or raw math spans
        latex = root.get("data-latex") if isinstance(root, Tag) else None
        if latex:
            return normalize_markdown(f"$${latex}$$") if "\n" in latex or len(latex) > 40 else normalize_markdown(f"${latex}$")

        math = root.find(class_=re.compile(r"math|formula|katex", re.I)) if isinstance(root, Tag) else None
        if math and math.get("data-latex"):
            latex = math["data-latex"]
            return normalize_markdown(f"${latex}$")

    converted = md(html, heading_style="ATX", bullets="-", strip=["script", "style"])
    return normalize_markdown(converted)


def extract_blocks_from_page_html(page_html: str) -> list[dict[str, str]]:
    """Find elements with data-block-id and return {id, html, tag}."""
    soup = BeautifulSoup(page_html, "lxml")
    blocks: list[dict[str, str]] = []
    for el in soup.find_all(attrs={"data-block-id": True}):
        bid = str(el["data-block-id"])
        blocks.append(
            {
                "id": bid,
                "tag": el.name or "div",
                "html": str(el),
                "markdown": html_fragment_to_markdown(str(el)),
            }
        )
    return blocks

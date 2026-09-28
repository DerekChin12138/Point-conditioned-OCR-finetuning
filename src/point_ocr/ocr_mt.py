"""Point OCR + English→Chinese translation supervision helpers."""

from __future__ import annotations

import re
from typing import Any

OCR_MT_PROMPT_KEYS = frozenset(
    {
        "ocr_mt",
        "ocr_mt_v1",
        "point_ocr_mt",
        "point_ocr_mt_v1",
        "POINT_OCR_MT_V1",
    }
)

_SOURCE_RE = re.compile(
    r"<source>\s*(.*?)\s*</source>",
    re.IGNORECASE | re.DOTALL,
)
_TRANS_RE = re.compile(
    r"<translation>\s*(.*?)\s*</translation>",
    re.IGNORECASE | re.DOTALL,
)


def is_ocr_mt_prompt(prompt_key: str | None) -> bool:
    if not prompt_key:
        return False
    key = str(prompt_key).strip()
    return key in OCR_MT_PROMPT_KEYS or key.lower() in {k.lower() for k in OCR_MT_PROMPT_KEYS}


def format_ocr_mt_target(source: str, translation: str) -> str:
    src = (source or "").strip()
    zh = (translation or "").strip()
    return f"<source>\n{src}\n</source>\n<translation>\n{zh}\n</translation>"


def parse_ocr_mt_target(text: str) -> tuple[str, str] | None:
    """Return (source, translation) if both tags are present."""
    raw = text or ""
    sm = _SOURCE_RE.search(raw)
    tm = _TRANS_RE.search(raw)
    if not sm or not tm:
        return None
    return sm.group(1).strip(), tm.group(1).strip()


def wrap_point_target(
    markdown: str,
    *,
    prompt_key: str | None,
    translation: str | None,
    is_negative: bool = False,
) -> str:
    """Empty stays empty. Positives become source/translation XML under OCR-MT prompts."""
    md = markdown or ""
    if is_negative or not md.strip():
        return ""
    if not is_ocr_mt_prompt(prompt_key):
        return md
    return format_ocr_mt_target(md, translation or "")


def normalize_pair_item(item: Any) -> dict[str, str] | None:
    """Accept a plain string or `{en, zh, ...}` dict."""
    if isinstance(item, str):
        en = item.strip()
        if not en:
            return None
        return {"en": en, "zh": "", "domain": "legacy", "source": "legacy"}
    if not isinstance(item, dict):
        return None
    en = str(item.get("en") or item.get("src") or item.get("text") or "").strip()
    zh = str(item.get("zh") or item.get("tgt") or item.get("translation") or "").strip()
    if not en:
        return None
    domain = str(item.get("domain") or "unknown").strip() or "unknown"
    source = str(item.get("source") or item.get("corpus") or "").strip()
    return {"en": en, "zh": zh, "domain": domain, "source": source}


def pair_norm_key(en: str) -> str:
    return " ".join((en or "").casefold().split())

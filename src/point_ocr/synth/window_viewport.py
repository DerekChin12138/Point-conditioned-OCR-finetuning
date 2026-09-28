"""Sample focused-window capture sizes for synth rendering.

Product assumption: the app screenshots the *currently focused OS window*
(not the whole desktop). Training images should therefore span a range of
client-area sizes similar to real resizable app windows — about **3:4 to 16:9**,
not phone 9:16 or ultrawide 21:9 strips.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WindowCaptureSpec:
    """Playwright context size + how the screenshot is taken."""

    viewport: tuple[int, int]  # CSS pixels (width, height)
    device_scale_factor: float = 1.0
    full_page: bool = False  # False ≡ visible window framebuffer
    family: str = "window"

    @property
    def width(self) -> int:
        return int(self.viewport[0])

    @property
    def height(self) -> int:
        return int(self.viewport[1])

    @property
    def aspect(self) -> float:
        return float(self.width) / max(1, float(self.height))

    @property
    def css_pixels(self) -> int:
        return int(self.width * self.height)

    @property
    def output_pixels(self) -> int:
        s = float(self.device_scale_factor)
        return int(self.width * self.height * s * s)

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        d["viewport"] = [self.width, self.height]
        d["aspect"] = round(self.aspect, 4)
        d["css_pixels"] = self.css_pixels
        d["output_pixels"] = self.output_pixels
        return d


# Natural focused-app windows only: ~3:4 reader / split through 16:9 monitor.
# No 21:9 strips, no 9:16 phone frames — those squash long documents.
_MIN_ASPECT = 3 / 4  # 0.75
_MAX_ASPECT = 16 / 9  # ≈1.778

_PRESET_WINDOWS: tuple[tuple[str, int, int], ...] = (
    # Compact / floating panels (still ~4:3)
    ("compact", 640, 480),
    ("compact", 720, 540),
    ("compact", 800, 600),
    ("compact", 900, 675),
    # Standard document / browser windows
    ("standard", 1024, 768),
    ("standard", 1280, 720),
    ("standard", 1280, 800),
    ("standard", 1366, 768),
    ("standard", 1440, 900),
    ("standard", 1536, 864),
    ("standard", 1600, 900),
    ("standard", 1680, 1050),
    ("standard", 1920, 1080),
    # Large / hi-dpi CSS viewports (before DPR)
    ("large", 2048, 1152),
    ("large", 2560, 1440),
    ("large", 1920, 1200),
    # Mild portrait: side-by-side / reader, not phone
    ("portrait", 720, 960),
    ("portrait", 800, 1064),
    ("portrait", 900, 1200),
    ("portrait", 960, 1280),
    # Near-square
    ("square", 800, 800),
    ("square", 1024, 1024),
    ("square", 1200, 1100),
)

# Aspect-ratio mixture for continuous sampling (width / height).
_ASPECT_MIX: tuple[tuple[str, float, float], ...] = (
    # (family, aspect, weight) — weight relative
    ("landscape_16_9", 16 / 9, 22),
    ("landscape_16_10", 16 / 10, 16),
    ("landscape_3_2", 3 / 2, 12),
    ("landscape_4_3", 4 / 3, 14),
    ("portrait_3_4", 3 / 4, 10),
    ("portrait_4_5", 4 / 5, 8),
    ("square", 1.0, 10),
    ("square_5_4", 5 / 4, 6),
)

# Device pixel ratio — product captures often include 1x and retina-ish buffers.
_DPR_CHOICES: tuple[tuple[float, float], ...] = (
    (1.0, 0.48),
    (1.25, 0.14),
    (1.5, 0.26),
    (2.0, 0.12),
)

# Soft caps so Unsloth max_pixels (~8.3M) still fits after DPR.
_MIN_SIDE = 400
_MAX_SIDE_CSS = 2880
_MAX_OUTPUT_PIXELS = 8_294_400
_MIN_CSS_PIXELS = 480 * 640
_MAX_CSS_PIXELS = 2560 * 1600


def _snap(v: float, multiple: int = 8) -> int:
    x = int(round(float(v) / multiple) * multiple)
    return max(multiple, x)


def _fit_aspect(w: int, h: int) -> tuple[int, int]:
    """Keep width/height inside [_MIN_ASPECT, _MAX_ASPECT] by adjusting height."""
    w = max(_MIN_SIDE, int(w))
    h = max(_MIN_SIDE, int(h))
    ar = float(w) / max(1.0, float(h))
    if ar < _MIN_ASPECT:
        h = _snap(w / _MIN_ASPECT)
    elif ar > _MAX_ASPECT:
        h = _snap(w / _MAX_ASPECT)
    h = max(_MIN_SIDE, min(_MAX_SIDE_CSS, h))
    # If height clamp re-broke the ratio, nudge width.
    ar = float(w) / max(1.0, float(h))
    if ar < _MIN_ASPECT:
        w = _snap(h * _MIN_ASPECT)
    elif ar > _MAX_ASPECT:
        w = _snap(h * _MAX_ASPECT)
    w = max(_MIN_SIDE, min(_MAX_SIDE_CSS, w))
    h = max(_MIN_SIDE, min(_MAX_SIDE_CSS, h))
    # Snap-to-8 can overshoot the band; one more pass.
    ar = float(w) / max(1.0, float(h))
    if ar < _MIN_ASPECT - 1e-6:
        h = min(_MAX_SIDE_CSS, max(_MIN_SIDE, _snap(w / _MIN_ASPECT)))
    elif ar > _MAX_ASPECT + 1e-6:
        h = min(_MAX_SIDE_CSS, max(_MIN_SIDE, _snap(w / _MAX_ASPECT)))
    return w, h


def _clamp_viewport(w: int, h: int, *, dpr: float) -> tuple[int, int]:
    w, h = _fit_aspect(w, h)
    w = int(max(_MIN_SIDE, min(_MAX_SIDE_CSS, w)))
    h = int(max(_MIN_SIDE, min(_MAX_SIDE_CSS, h)))
    # Shrink until output pixels fit the training soft cap.
    out = w * h * dpr * dpr
    guard = 0
    while out > _MAX_OUTPUT_PIXELS and guard < 40:
        scale = math.sqrt(_MAX_OUTPUT_PIXELS / out) * 0.98
        w = max(_MIN_SIDE, _snap(w * scale))
        h = max(_MIN_SIDE, _snap(h * scale))
        out = w * h * dpr * dpr
        guard += 1
    return _fit_aspect(w, h)


def _sample_dpr(rng: random.Random) -> float:
    vals = [v for v, _ in _DPR_CHOICES]
    weights = [w for _, w in _DPR_CHOICES]
    return float(rng.choices(vals, weights=weights, k=1)[0])


def _sample_preset(rng: random.Random) -> tuple[str, int, int]:
    family, w, h = rng.choice(_PRESET_WINDOWS)
    # Mild jitter so presets are not a tiny discrete set.
    jitter = 1.0 + rng.uniform(-0.06, 0.06)
    w = _snap(w * jitter)
    h = _snap(h * jitter)
    return family, w, h


def _sample_aspect_area(rng: random.Random) -> tuple[str, int, int]:
    families = [f for f, _, _ in _ASPECT_MIX]
    aspects = [a for _, a, _ in _ASPECT_MIX]
    weights = [w for _, _, w in _ASPECT_MIX]
    idx = rng.choices(range(len(families)), weights=weights, k=1)[0]
    family = families[idx]
    ar = aspects[idx] * (1.0 + rng.uniform(-0.03, 0.03))
    ar = max(_MIN_ASPECT, min(_MAX_ASPECT, ar))

    # Log-uniform CSS pixel area.
    lo, hi = math.log(_MIN_CSS_PIXELS), math.log(_MAX_CSS_PIXELS)
    area = math.exp(rng.uniform(lo, hi))
    h = math.sqrt(area / ar)
    w = ar * h
    return family, _snap(w), _snap(h)


def sample_window_capture(
    rng: random.Random | None = None,
    *,
    full_page_prob: float = 0.0,
    min_css_pixels: int | None = None,
    min_aspect: float | None = None,
) -> WindowCaptureSpec:
    """Sample a focused-window capture configuration.

    - ``full_page=False`` (default): screenshot equals the window framebuffer
      (product-aligned). Long documents may be cropped; builders drop partial GT.
    - ``full_page=True`` is opt-in only: Playwright extends height with content
      and can produce extreme portrait images, so it is off by default.
    - ``min_css_pixels`` / ``min_aspect``: pool-specific floors (e.g. multi-column
      wrap needs a wide enough window or every fragment is cropped).
    """
    rng = rng or random.Random()
    area_floor = int(min_css_pixels) if min_css_pixels else 0
    aspect_floor = float(min_aspect) if min_aspect else 0.0
    last: WindowCaptureSpec | None = None
    for _ in range(24):
        dpr = _sample_dpr(rng)

        if rng.random() < 0.45:
            family, w, h = _sample_preset(rng)
        else:
            family, w, h = _sample_aspect_area(rng)

        full_page = bool(rng.random() < float(full_page_prob))
        if full_page:
            # Width still diverse; height is a "window chrome" floor — Playwright
            # full_page extends beyond it when content is taller.
            family = f"{family}_scroll"
            # Prefer a moderate height floor so short pages don't force huge empty canvas.
            # Keep it inside the natural aspect band (no squat ultrawide floors).
            h_lo = max(_MIN_SIDE, _snap(w / _MAX_ASPECT))
            h_hi = max(h_lo, min(h, _snap(w / _MIN_ASPECT), _snap(1100)))
            h = _snap(rng.uniform(h_lo, h_hi)) if h_hi > h_lo else h_lo

        w, h = _clamp_viewport(w, h, dpr=dpr)
        last = WindowCaptureSpec(
            viewport=(w, h),
            device_scale_factor=dpr,
            full_page=full_page,
            family=family,
        )
        if area_floor and last.css_pixels < area_floor:
            continue
        if aspect_floor and last.aspect + 1e-6 < aspect_floor:
            continue
        return last

    assert last is not None
    if area_floor and last.css_pixels < area_floor:
        scale = math.sqrt(area_floor / max(1, last.css_pixels))
        w, h = _clamp_viewport(_snap(last.width * scale), _snap(last.height * scale), dpr=last.device_scale_factor)
        last = WindowCaptureSpec(
            viewport=(w, h),
            device_scale_factor=last.device_scale_factor,
            full_page=last.full_page,
            family=last.family,
        )
    return last

#!/usr/bin/env python3
"""Local playground: click-to-place x45b marker, tweak size/alpha/decode, run a LoRA.

  uv run python eval/marker_studio/server.py
  uv run python eval/marker_studio/server.py --adapter checkpoints/<run>/adapter_final
  # open http://127.0.0.1:7865
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.image_resize import resize_for_ovis  # noqa: E402
from point_ocr.infer import DEFAULT_POINT_MAX_NEW_TOKENS, generate_point_text  # noqa: E402
from point_ocr.marker import (  # noqa: E402
    CURRENT_MARKER_TAG,
    MARKER_ALPHA_AT_REF,
    MARKER_ALPHA_MIN,
    MARKER_AREA_FRAC,
    MARKER_AREA_FRAC_LEGACY,
    MARKER_AREA_FRAC_MAX,
    MARKER_AREA_FRAC_MIN,
    MARKER_AREA_SCALE_LARGE,
    MARKER_AREA_SCALE_SMALL,
    MARKER_REF_ARM_PX,
    MARKER_REF_IMAGE_WH,
    MARKER_REF_SQUARE_AREA,
    draw_crosshair,
    spec_for_image,
)
from point_ocr.prompts import (  # noqa: E402
    PAGE_PROMPT,
    POINT_PROMPT,
    POINT_PROMPT_A1_V2,
    POINT_PROMPT_A2_V2,
    POINT_PROMPT_A2_V3,
    POINT_PROMPT_OCR_MT_V1,
    POINT_PROMPT_COORD_PROBE_TEMPLATE,
    POINT_PROMPT_COORD_Q1_TEMPLATE,
    POINT_PROMPT_COORD_V1_TEMPLATE,
    POINT_PROMPT_RED_REGION,
    apply_coord_placeholders,
    format_coord_prompt_q1,
    format_point_2d_json,
    pixel_to_pct_topleft,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from view_ops import apply_studio_view, remap_point  # noqa: E402
from quad_probe import (  # noqa: E402
    PROBE_POINTS,
    expected_color,
    interpret_probe,
    make_quadrant_image,
)

INDEX = HERE / "index.html"
LABEL = HERE / "label.html"
SAMPLES = ROOT / "data/real_world_sample"
GALLERIES: dict[str, Path] = {
    "real": ROOT / "data/real_world_sample",
    "q1_test": ROOT / "data/studio_galleries/q1_test",
    "signal_multi_frag": ROOT / "data/studio_galleries/signal_multi_frag",
    "ood": ROOT / "data/studio_galleries/ood",
    "probe": ROOT / "data/studio_galleries/probe",
    "inbox": ROOT / "data/label_inbox",
}
LABEL_INBOX = ROOT / "data/label_inbox"
LABEL_OUT = ROOT / "data/label_out"
LABEL_WORK = ROOT / "data/label_work"
DEFAULT_LORA: Path | None = ROOT / "checkpoints/q2_2b/adapter_final"
# Studio inference resolution. Must match training `--max-pixels` or the probe is
# out-of-distribution: training used to be silently capped at 768px wide by the
# Unsloth collator, while the studio/evals used full resolution (see 2026-09-28 fix).
RESIZE_MAX_PIXELS = 2048 * 2048
DEFAULT_BASE = str((ROOT / "models" / "Qwen3.5-0.8B").resolve())
BASE_OVIS = "ATH-MaaS/OvisOCR2"
QWEN35_LOCAL = ROOT / "models" / "Qwen3.5-0.8B"
HOST = "127.0.0.1"
PORT = 7865
PROMPT_MAX_CHARS = 8000
# Annotation workbench: one opened real-screenshot folder (not studio galleries).
_LABEL_WORK: dict[str, Any] = {"root": None}
PROMPT_PRESETS = [
    {"id": "ocr_mt_v1", "label": "ocr_mt_v1 · 点OCR + 英译中", "text": POINT_PROMPT_OCR_MT_V1},
    {"id": "a2_v3", "label": "a2_v3 · Q1 打标", "text": POINT_PROMPT_A2_V3},
    {"id": "coord_q1", "label": "coord_q1 · 像素 point_2d（不画叉）", "text": POINT_PROMPT_COORD_Q1_TEMPLATE},
    {"id": "coord_probe", "label": "coord_probe · 四象限颜色（不声明原点）", "text": POINT_PROMPT_COORD_PROBE_TEMPLATE},
    {"id": "red_region", "label": "red_region · 红色区域范围", "text": POINT_PROMPT_RED_REGION},
    {"id": "coord_v1", "label": "coord_v1 · 旧 [0,1000) 草稿", "text": POINT_PROMPT_COORD_V1_TEMPLATE},
    {"id": "a2_v2", "label": "a2_v2 · B1 Ovis", "text": POINT_PROMPT_A2_V2},
    {"id": "a1_v2", "label": "a1_v2 · 最小单块", "text": POINT_PROMPT_A1_V2},
    {"id": "page", "label": "PAGE · 整页 Markdown", "text": PAGE_PROMPT},
]

_STATE: dict[str, Any] = {"model": None, "tokenizer": None, "base": None, "lora": None}
_JOBS: queue.Queue[tuple[threading.Event, dict[str, Any], dict[str, Any]]] = queue.Queue()


def _is_lora_dir(spec: str) -> bool:
    p = Path(spec)
    return p.is_dir() and (p / "adapter_config.json").is_file()


def _lora_base_model(spec: str) -> str:
    """Read PEFT base_model_name_or_path so a LoRA is not loaded onto the wrong base."""
    cfg_path = Path(spec) / "adapter_config.json"
    if not cfg_path.is_file():
        return ""
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    base = str(cfg.get("base_model_name_or_path") or "").strip()
    if not base:
        return ""
    p = Path(base)
    if p.is_dir():
        return str(p.resolve())
    alt = (ROOT / p).resolve()
    if alt.is_dir():
        return str(alt)
    return base


def _same_model_id(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    pa, pb = Path(a), Path(b)
    try:
        if pa.exists() and pb.exists() and pa.resolve() == pb.resolve():
            return True
    except OSError:
        pass
    return pa.name == pb.name


def _qwen35_path() -> str:
    if (QWEN35_LOCAL / "config.json").is_file():
        return str(QWEN35_LOCAL.resolve())
    return "Qwen/Qwen3.5-0.8B"


def _is_full_vl_checkpoint(path: Path) -> bool:
    """A merged / standalone VL dir (not a LoRA adapter folder)."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    if (path / "adapter_config.json").is_file():
        return False
    return (
        (path / "model.safetensors").is_file()
        or (path / "pytorch_model.bin").is_file()
        or any(path.glob("model-*.safetensors"))
    )


def _list_bases() -> list[dict[str, str]]:
    rows = [
        {"id": "Qwen3.5-0.8B", "path": _qwen35_path()},
        {"id": "OvisOCR2", "path": BASE_OVIS},
    ]
    ckpt = ROOT / "checkpoints"
    if ckpt.is_dir():
        for p in sorted(ckpt.iterdir()):
            if _is_full_vl_checkpoint(p):
                rows.append({"id": p.name, "path": str(p.resolve())})
    return rows


def _list_loras() -> list[dict[str, str]]:
    rows = [{"id": "（无 LoRA）", "path": "", "expected_base": ""}]
    ckpt = ROOT / "checkpoints"
    if ckpt.is_dir():
        for p in sorted(ckpt.glob("*/adapter_final")):
            rows.append(
                {
                    "id": p.parent.name,
                    "path": str(p.resolve()),
                    "expected_base": _lora_base_model(str(p)),
                }
            )
    return rows


def _list_image_files(root: Path, *, recursive: bool) -> list[Path]:
    if not root.is_dir():
        return []
    it = root.rglob("*") if recursive else root.glob("*")
    rows: list[Path] = []
    for p in it:
        if not p.is_file() or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        # skip Q1 marked jpgs / sidecar-only dirs
        if p.parent.name in {"marked", "marked_nochrome"}:
            continue
        rows.append(p)
    return sorted(rows)


def _list_samples() -> list[dict[str, str]]:
    """Gallery entries for the dropdown. id is gallery-relative and URL-safe."""
    groups = [
        ("real", GALLERIES["real"], "真实截图", False),
        ("q1_test", GALLERIES["q1_test"], "Q1 test 合成页（未打标）", False),
        ("signal_multi_frag", GALLERIES["signal_multi_frag"], "Signal val · multi_frag", False),
        ("ood", GALLERIES["ood"], "OOD 探针", True),
        ("probe", GALLERIES["probe"], "坐标先验探针", False),
        ("inbox", GALLERIES["inbox"], "待标注 inbox", True),
    ]
    out: list[dict[str, str]] = []
    for gid, root, label, recursive in groups:
        if gid == "real":
            files = _list_image_files(root, recursive=False)
            nc = root / "nochrome"
            files += _list_image_files(nc, recursive=False)
        else:
            files = _list_image_files(root, recursive=recursive)
        for p in files:
            rel = p.relative_to(root).as_posix()
            out.append({"id": f"{gid}/{rel}", "label": rel, "group": label})
    return out


def _resolve_sample(rel: str) -> Path:
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel:
        raise FileNotFoundError(rel)
    gallery = None
    rest = rel
    if "/" in rel:
        head, tail = rel.split("/", 1)
        if head in GALLERIES:
            gallery, rest = head, tail
    if ".." in Path(rest).parts:
        raise FileNotFoundError(rel)
    if gallery is None:
        root = SAMPLES.resolve()
        fp = root / rest
    else:
        root = GALLERIES[gallery].resolve()
        fp = root / rest
    # Galleries may symlink out to pool renders; only check the joined path stays
    # under root (do not require the symlink *target* to stay inside).
    try:
        if not fp.is_relative_to(root):
            raise FileNotFoundError(rel)
    except (ValueError, AttributeError):
        if not str(fp).startswith(str(root)):
            raise FileNotFoundError(rel) from None
    if not fp.is_file():
        raise FileNotFoundError(rel)
    return fp


def _load_model(base: str, lora: str = "") -> None:
    import torch
    from unsloth import FastVisionModel

    base = str(base or "").strip() or _qwen35_path()
    lora = str(lora or "").strip()
    if lora and not _is_lora_dir(lora):
        raise ValueError(f"not a LoRA dir (missing adapter_config.json): {lora}")
    if lora:
        expected = _lora_base_model(lora)
        if expected and not _same_model_id(base, expected):
            raise ValueError(
                f"LoRA {Path(lora).parent.name} expects base {expected}, "
                f"but you selected {base}. Change the base dropdown (or pick no LoRA)."
            )
    if (
        _STATE["model"] is not None
        and _STATE["base"] == base
        and _STATE["lora"] == lora
    ):
        return
    _STATE["model"] = None
    _STATE["tokenizer"] = None
    _STATE["base"] = None
    _STATE["lora"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    import os

    local_base = Path(base)
    is_local = local_base.is_dir()
    # Unsloth pings the Hub after its banner even for a local dir and the
    # generate worker looks frozen (no "ready") until that times out.
    prev_hub = os.environ.get("HF_HUB_OFFLINE")
    prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    if is_local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    load_kwargs = dict(
        load_in_4bit=False,
        dtype=dtype,
        use_gradient_checkpointing="unsloth",
    )
    if is_local:
        load_kwargs["local_files_only"] = True
    print(f"loading base={base} lora={lora or '(none)'} dtype={dtype}", flush=True)
    try:
        model, tok = FastVisionModel.from_pretrained(base, **load_kwargs)
    finally:
        if is_local:
            if prev_hub is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_hub
            if prev_tf is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = prev_tf
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora)
    FastVisionModel.for_inference(model)
    _STATE["model"] = model
    _STATE["tokenizer"] = tok
    _STATE["base"] = base
    _STATE["lora"] = lora
    print("ready", flush=True)


def _pair_from_body(body: dict[str, Any]) -> tuple[str, str]:
    """Resolve base + optional LoRA. Legacy `adapter` still accepted."""
    base = str(body.get("base") or "").strip()
    lora = str(body.get("lora") or "").strip()
    legacy = str(body.get("adapter") or "").strip()
    if legacy and not base and not lora:
        if _is_lora_dir(legacy):
            lora = legacy
            base = _lora_base_model(legacy) or _qwen35_path()
        else:
            base = legacy
    if not base:
        base = str(DEFAULT_BASE)
    if not lora and not legacy and DEFAULT_LORA and DEFAULT_LORA.is_dir() and body.get("use_default_lora"):
        lora = str(DEFAULT_LORA)
    return base, lora


def _b64_to_image(data_url: str) -> Image.Image:
    raw = data_url.split(",", 1)[-1]
    blob = base64.b64decode(raw)
    return Image.open(io.BytesIO(blob)).convert("RGB")


def _load_sample(rel: str) -> Image.Image:
    return Image.open(_resolve_sample(rel)).convert("RGB")


def _label_root() -> Path | None:
    root = _LABEL_WORK.get("root")
    return Path(root) if root else None


def _list_label_files(root: Path) -> list[dict[str, str]]:
    files = _list_image_files(root, recursive=True)
    out: list[dict[str, str]] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        out.append({"id": rel, "label": rel, "path": str(p.resolve())})
    return out


def _normalize_dir_path(raw: str) -> Path:
    """Accept POSIX, ~/…, or Windows D:\\foo → /mnt/d/foo under WSL."""
    s = str(raw or "").strip().strip('"').strip("'")
    if not s:
        raise ValueError("empty path")
    # Windows drive letter → WSL mount
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/").lstrip("/")
        s = f"/mnt/{drive}/{rest}"
    else:
        s = s.replace("\\", "/")
    return Path(s).expanduser().resolve()


def _open_label_dir(raw: str) -> dict[str, Any]:
    p = _normalize_dir_path(raw)
    if not p.is_dir():
        raise ValueError(f"not a directory: {p}")
    _LABEL_WORK["root"] = p
    files = _list_label_files(p)
    return {"ok": True, "root": str(p), "n": len(files), "files": files}


def _import_label_files(files: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Save browser-picked folder files under data/label_work/<stamp>/."""
    if not files:
        raise ValueError("no files uploaded")
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = (LABEL_WORK / f"import_{stamp}").resolve()
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for rel, blob in files:
        rel = rel.replace("\\", "/").lstrip("/")
        # webkitRelativePath often "FolderName/sub/a.png" — drop top folder name
        parts = [p for p in rel.split("/") if p and p != ".."]
        if len(parts) >= 2:
            parts = parts[1:]
        if not parts:
            continue
        if Path(parts[-1]).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        out = dest.joinpath(*parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        n += 1
    if n == 0:
        raise ValueError("no image files in selection")
    return _open_label_dir(str(dest))


def _resolve_label_file(rel: str) -> Path:
    root = _label_root()
    if root is None:
        raise FileNotFoundError("no label folder open; call /api/label/open_dir first")
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/"):
        raise FileNotFoundError(rel)
    fp = (root / rel).resolve()
    if not str(fp).startswith(str(root.resolve())) or not fp.is_file():
        raise FileNotFoundError(rel)
    return fp


def _image_from_body(body: dict[str, Any]) -> Image.Image:
    """Load image for OCR / label: label_rel, sample gallery id, or base64."""
    label_rel = str(body.get("label_rel") or "").strip()
    if label_rel:
        return Image.open(_resolve_label_file(label_rel)).convert("RGB")
    if body.get("sample"):
        return _load_sample(str(body["sample"]))
    return _b64_to_image(str(body["image"]))


def _decode_kwargs(body: dict[str, Any]) -> dict[str, Any]:
    """Clamp studio decode knobs. n>1 implies sampling (for RL-support probes)."""
    temperature = float(body.get("temperature") if body.get("temperature") not in (None, "") else 0)
    do_sample = bool(body.get("do_sample"))
    n = max(1, min(8, int(body.get("n") or 1)))
    if n > 1:
        do_sample = True
        if temperature <= 0:
            temperature = 0.7
    if temperature <= 0:
        do_sample = False
        n = 1
    seed_raw = body.get("seed")
    seed = None if seed_raw in (None, "") else int(seed_raw)
    top_k_raw = body.get("top_k")
    top_k = 0 if top_k_raw in (None, "") else int(top_k_raw)
    return {
        "do_sample": do_sample,
        "temperature": temperature if do_sample else 0.0,
        "top_p": min(1.0, max(0.05, float(body.get("top_p") or 0.9))),
        "top_k": top_k if top_k > 0 else None,
        "repetition_penalty": float(body.get("repetition_penalty") or 1.0),
        "max_new_tokens": max(16, min(1024, int(body.get("max_new_tokens") or DEFAULT_POINT_MAX_NEW_TOKENS))),
        "n": n,
        "seed": seed,
    }


def _resolve_area_frac(body: dict[str, Any]) -> float:
    """Explicit absolute ``area_frac`` wins; else legacy ``area_scale``; else default."""
    raw = body.get("area_frac")
    if raw not in (None, ""):
        return float(raw)
    scale = body.get("area_scale")
    if scale not in (None, ""):
        return MARKER_AREA_FRAC_LEGACY * float(scale)
    return MARKER_AREA_FRAC


def _generate(body: dict[str, Any]) -> dict[str, Any]:
    base, lora = _pair_from_body(body)
    _load_model(base, lora)
    sample = body.get("sample")
    if sample:
        img = _load_sample(str(sample))
    else:
        img = _b64_to_image(str(body["image"]))
    src_wh = list(img.size)
    img, ox, oy = apply_studio_view(
        img, crop=body.get("crop"), blur=body.get("blur")
    )
    x, y = remap_point(float(body["x"]), float(body["y"]), ox, oy)
    if not (0 <= x < img.size[0] and 0 <= y < img.size[1]):
        raise ValueError(
            f"point ({body.get('x')}, {body.get('y')}) is outside the crop"
        )
    scale = float(body.get("scale") or 1.0)
    alpha_raw = body.get("alpha")
    alpha = None if alpha_raw in (None, "", "auto") else int(alpha_raw)
    area_frac = _resolve_area_frac(body)
    spec = spec_for_image(*img.size, area_frac=area_frac, scale=scale, alpha=alpha)
    skip_marker = bool(body.get("skip_marker") or body.get("no_marker"))
    stamped = img if skip_marker else draw_crosshair(img, x, y, spec)
    prompt_raw = body.get("prompt")
    prompt = str(prompt_raw).strip() if prompt_raw is not None else ""
    if not prompt:
        prompt = POINT_PROMPT
    xi, yi = int(round(x)), int(round(y))
    if "{point}" in prompt and ("point_2d" in prompt or "pixel" in prompt.lower()):
        prompt = prompt.replace("{point}", format_point_2d_json(xi, yi))
    else:
        xp, yp = pixel_to_pct_topleft(x, y, img.size[0], img.size[1])
        prompt = apply_coord_placeholders(prompt, xp, yp)
    if len(prompt) > PROMPT_MAX_CHARS:
        prompt = prompt[:PROMPT_MAX_CHARS]
    vis = resize_for_ovis(stamped, max_pixels=RESIZE_MAX_PIXELS)
    dec = _decode_kwargs(body)
    preds: list[dict[str, Any]] = []
    for i in range(int(dec["n"])):
        seed = None if dec["seed"] is None else int(dec["seed"]) + i
        out = generate_point_text(
            _STATE["model"],
            _STATE["tokenizer"],
            vis,
            prompt,
            point=[x, y],
            clean=True,
            do_sample=dec["do_sample"],
            temperature=dec["temperature"],
            top_p=dec["top_p"],
            top_k=dec["top_k"],
            repetition_penalty=dec["repetition_penalty"],
            max_new_tokens=dec["max_new_tokens"],
            seed=seed,
        )
        text = str(out.cleaned) if hasattr(out, "cleaned") else str(out)
        raw = str(out.raw) if hasattr(out, "raw") else text
        leak = bool(getattr(out, "format_leak", False))
        preds.append({"text": text, "raw": raw, "format_leak": leak, "seed": seed})
    buf = io.BytesIO()
    stamped.save(buf, format="JPEG", quality=88)
    stamped_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return {
        "pred": preds[0]["text"] if preds else "",
        "preds": preds,
        "gen": dec,
        "spec": spec.to_dict(),
        "base": base,
        "lora": lora,
        "adapter": lora or base,
        "stamped": stamped_b64,
        "image_wh": list(img.size),
        "source_wh": src_wh,
        "view_origin": [ox, oy],
        "arm_px": spec.cross_half_length_px * 2,
        "alpha": spec.cross_color[3],
        "prompt": prompt,
        "skip_marker": skip_marker,
        "point_pct": list(pixel_to_pct_topleft(x, y, img.size[0], img.size[1])),
        "point_2d": [xi, yi],
    }


def _coord_probe(body: dict[str, Any]) -> dict[str, Any]:
    base, lora = _pair_from_body(body)
    _load_model(base, lora)
    img = make_quadrant_image(800)
    vis = resize_for_ovis(img, max_pixels=RESIZE_MAX_PIXELS)
    w, h = vis.size
    rows: list[dict[str, Any]] = []
    for x_pct, y_pct in PROBE_POINTS:
        px = (x_pct / 100.0) * w
        py = (y_pct / 100.0) * h
        prompt = apply_coord_placeholders(POINT_PROMPT_COORD_PROBE_TEMPLATE, x_pct, y_pct)
        out = generate_point_text(
            _STATE["model"],
            _STATE["tokenizer"],
            vis,
            prompt,
            point=[px, py],
            clean=True,
            do_sample=False,
            max_new_tokens=16,
        )
        text = str(out.cleaned) if hasattr(out, "cleaned") else str(out)
        rows.append(
            {
                "x_pct": x_pct,
                "y_pct": y_pct,
                "pred": text,
                "expected_topleft": expected_color(x_pct, y_pct, origin="top-left"),
                "expected_bottomleft": expected_color(x_pct, y_pct, origin="bottom-left"),
            }
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "base": base,
        "lora": lora,
        "legend": {
            "top-left": "red",
            "top-right": "green",
            "bottom-left": "blue",
            "bottom-right": "yellow",
        },
        "results": rows,
        "interpretation": interpret_probe(rows),
        "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
    }


def _red_region_probe(body: dict[str, Any]) -> dict[str, Any]:
    """Open-ended: how does the model choose to write the red region's extent?"""
    base, lora = _pair_from_body(body)
    _load_model(base, lora)
    img = make_quadrant_image(800)
    vis = resize_for_ovis(img, max_pixels=RESIZE_MAX_PIXELS)
    prompt = POINT_PROMPT_RED_REGION
    out = generate_point_text(
        _STATE["model"],
        _STATE["tokenizer"],
        vis,
        prompt,
        clean=True,
        do_sample=False,
        max_new_tokens=256,
    )
    text = str(out.cleaned) if hasattr(out, "cleaned") else str(out)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "base": base,
        "lora": lora,
        "prompt": prompt,
        "results": [
            {
                "color": "red",
                "pred": text,
                "note": "format is the signal: 0–1000 boxes, pixels, %, XML, etc.",
                "prompt": prompt,
            }
        ],
        "interpretation": (
            "Look at the notation, not the compass word. "
            "Pixel-space truth: red occupies the top-left half (about 0–50% width and 0–50% height if origin is top-left)."
        ),
        "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
    }


def _ocr_crop(body: dict[str, Any]) -> dict[str, Any]:
    """PAGE-style OCR on a crop (or full image) for the label tool."""
    base, lora = _pair_from_body(body)
    _load_model(base, lora)
    img = _image_from_body(body)
    box = body.get("bbox")
    if box and len(box) == 4:
        x1, y1, x2, y2 = [int(float(v)) for v in box]
        x1, x2 = sorted((max(0, x1), min(img.size[0], x2)))
        y1, y2 = sorted((max(0, y1), min(img.size[1], y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError("bbox too small")
        img = img.crop((x1, y1, x2, y2))
    prompt = str(body.get("prompt") or PAGE_PROMPT)
    vis = resize_for_ovis(img, max_pixels=RESIZE_MAX_PIXELS)
    out = generate_point_text(
        _STATE["model"],
        _STATE["tokenizer"],
        vis,
        prompt,
        clean=True,
        do_sample=False,
        max_new_tokens=max(64, min(2048, int(body.get("max_new_tokens") or 1024))),
    )
    text = str(out.cleaned) if hasattr(out, "cleaned") else str(out)
    return {"text": text, "base": base, "lora": lora, "crop_wh": list(img.size)}


def _worker() -> None:
    while True:
        event, body, box = _JOBS.get()
        try:
            kind = str(body.get("_job") or "generate")
            if kind == "coord_probe":
                box["ok"] = _coord_probe(body)
            elif kind == "red_region":
                box["ok"] = _red_region_probe(body)
            elif kind == "ocr":
                box["ok"] = _ocr_crop(body)
            else:
                box["ok"] = _generate(body)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            box["err"] = f"{type(e).__name__}: {e}"
        event.set()


def _run_job(body: dict[str, Any], *, timeout: float | None = None) -> tuple[int, dict[str, Any]]:
    event = threading.Event()
    box: dict[str, Any] = {}
    _JOBS.put((event, body, box))
    n = 1
    try:
        n = max(1, min(8, int(body.get("n") or 1)))
    except (TypeError, ValueError):
        n = 1
    if timeout is None:
        timeout = 180 + 90 * n
        if str(body.get("_job")) == "coord_probe":
            timeout = 360
    if not event.wait(timeout=timeout):
        return 504, {"error": "generate timed out"}
    if "err" in box:
        return 500, {"error": box["err"]}
    return 200, box["ok"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, raw, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"/", "/index.html"}:
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            return
        if path in {"/label", "/label.html"}:
            self._send(200, LABEL.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/label/list":
            root = _label_root()
            if root is None:
                self._json(200, {"root": None, "n": 0, "files": []})
                return
            files = _list_label_files(root)
            self._json(200, {"root": str(root), "n": len(files), "files": files})
            return
        if path.startswith("/label_files/"):
            rel = path[len("/label_files/") :]
            try:
                fp = _resolve_label_file(rel)
            except FileNotFoundError as e:
                self._json(404, {"error": str(e)})
                return
            data = fp.read_bytes()
            ctype = "image/png"
            if fp.suffix.lower() in {".jpg", ".jpeg"}:
                ctype = "image/jpeg"
            elif fp.suffix.lower() == ".webp":
                ctype = "image/webp"
            self._send(200, data, ctype)
            return
        if path == "/api/meta":
            default_lora = str(DEFAULT_LORA.resolve()) if DEFAULT_LORA and DEFAULT_LORA.is_dir() else ""
            self._json(
                200,
                {
                    "tag": CURRENT_MARKER_TAG,
                    "area_frac": MARKER_AREA_FRAC,
                    "area_frac_legacy": MARKER_AREA_FRAC_LEGACY,
                    "area_frac_min": MARKER_AREA_FRAC_MIN,
                    "area_frac_max": MARKER_AREA_FRAC_MAX,
                    "area_scale_large": MARKER_AREA_SCALE_LARGE,
                    "area_scale_small": MARKER_AREA_SCALE_SMALL,
                    "area_frac_presets": [
                        {"id": "min", "label": "新范围下限 0.08%", "area_frac": MARKER_AREA_FRAC_MIN},
                        {"id": "mid", "label": "新范围中点 0.14%", "area_frac": MARKER_AREA_FRAC},
                        {"id": "max", "label": "新范围上限 0.2%", "area_frac": MARKER_AREA_FRAC_MAX},
                        {"id": "x45d", "label": "旧 x45d 0.425%", "area_scale": MARKER_AREA_SCALE_LARGE},
                        {"id": "x45c", "label": "旧 x45c 0.5%", "area_scale": 1.0},
                    ],
                    "alpha_at_ref": MARKER_ALPHA_AT_REF,
                    "alpha_min": MARKER_ALPHA_MIN,
                    "ref_wh": list(MARKER_REF_IMAGE_WH),
                    "ref_arm_px": MARKER_REF_ARM_PX,
                    "ref_square_area": MARKER_REF_SQUARE_AREA,
                    "default_base": str(DEFAULT_BASE),
                    "default_lora": default_lora,
                    "bases": _list_bases(),
                    "loras": _list_loras(),
                    "samples": _list_samples(),
                    "loaded_base": _STATE["base"],
                    "loaded_lora": _STATE["lora"],
                    "prompt": POINT_PROMPT,
                    "prompt_presets": PROMPT_PRESETS,
                    "decode_defaults": {
                        "do_sample": False,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 0,
                        "repetition_penalty": 1.0,
                        "max_new_tokens": DEFAULT_POINT_MAX_NEW_TOKENS,
                        "n": 1,
                    },
                },
            )
            return
        if path.startswith("/samples/"):
            rel = path[len("/samples/") :]
            try:
                fp = _resolve_sample(rel)
            except FileNotFoundError:
                self._json(404, {"error": "sample not found"})
                return
            ext = fp.suffix.lower()
            ctype = "image/png" if ext == ".png" else "image/jpeg"
            if ext == ".webp":
                ctype = "image/webp"
            self._send(200, fp.read_bytes(), ctype)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        if parsed.path == "/api/generate":
            code, obj = _run_job(body)
            self._json(code, obj)
            return
        if parsed.path == "/api/coord_probe":
            body["_job"] = "coord_probe"
            code, obj = _run_job(body)
            self._json(code, obj)
            return
        if parsed.path == "/api/red_region_probe":
            body["_job"] = "red_region"
            code, obj = _run_job(body, timeout=240)
            self._json(code, obj)
            return
        if parsed.path == "/api/ocr":
            body["_job"] = "ocr"
            code, obj = _run_job(body, timeout=240)
            self._json(code, obj)
            return
        if parsed.path == "/api/label/open_dir":
            try:
                self._json(200, _open_label_dir(str(body.get("path") or "")))
            except (OSError, ValueError) as e:
                self._json(400, {"error": str(e)})
            return
        if parsed.path == "/api/label/import_files":
            # body: { "files": [ {"rel": "a/b.png", "data_url": "data:image/...;base64,..." }, ... ] }
            try:
                rows = body.get("files") or []
                decoded: list[tuple[str, bytes]] = []
                for row in rows:
                    rel = str(row.get("rel") or row.get("name") or "")
                    raw = str(row.get("data_url") or row.get("data") or "")
                    if not rel or not raw:
                        continue
                    blob = base64.b64decode(raw.split(",", 1)[-1])
                    decoded.append((rel, blob))
                self._json(200, _import_label_files(decoded))
            except (OSError, ValueError) as e:
                self._json(400, {"error": str(e)})
            return
        if parsed.path == "/api/label/save":
            LABEL_OUT.mkdir(parents=True, exist_ok=True)
            name = str(body.get("name") or "ann").strip()
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:120]
            if not safe:
                safe = "ann"
            path = LABEL_OUT / f"{safe}.json"
            from datetime import datetime, timezone

            boxes_in = body.get("boxes") or []
            boxes_out: list[dict[str, Any]] = []
            for i, row in enumerate(boxes_in):
                text = str(row.get("text") or "")
                unit_id = str(row.get("unit_id") or f"u{i + 1}").strip() or f"u{i + 1}"
                boxes_out.append(
                    {
                        "bbox": row.get("bbox"),
                        "text": text,
                        "manual": bool(text.strip()),
                        "unit_id": unit_id,
                    }
                )
            unit_ids = sorted({b["unit_id"] for b in boxes_out})
            payload = {
                "name": safe,
                "source_path": body.get("source_path") or None,
                "source_rel": body.get("source_rel") or None,
                "label_root": body.get("label_root") or (str(_label_root()) if _label_root() else None),
                "image_wh": body.get("image_wh"),
                "boxes": boxes_out,
                "n_boxes": len(boxes_out),
                "n_units": len(unit_ids),
                "n_manual": sum(1 for b in boxes_out if b["manual"]),
                "n_pending_ocr": sum(1 for b in boxes_out if not b["manual"]),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._json(
                200,
                {
                    "ok": True,
                    "path": str(path),
                    "n_boxes": len(boxes_out),
                    "n_units": len(unit_ids),
                },
            )
            return
        if parsed.path == "/api/spec":
            w = int(body.get("w") or 1)
            h = int(body.get("h") or 1)
            scale = float(body.get("scale") or 1.0)
            alpha_raw = body.get("alpha")
            alpha = None if alpha_raw in (None, "", "auto") else int(alpha_raw)
            area_frac = _resolve_area_frac(body)
            spec = spec_for_image(w, h, area_frac=area_frac, scale=scale, alpha=alpha)
            self._json(
                200,
                {
                    "spec": spec.to_dict(),
                    "arm_px": spec.cross_half_length_px * 2,
                    "alpha": spec.cross_color[3],
                },
            )
            return
        self._json(404, {"error": "not found"})


def _ensure_probe_image() -> None:
    probe_dir = GALLERIES["probe"]
    probe_dir.mkdir(parents=True, exist_ok=True)
    fp = probe_dir / "quadrants.png"
    if not fp.is_file():
        make_quadrant_image(800).save(fp, format="PNG")


def main() -> None:
    global DEFAULT_LORA, DEFAULT_BASE, RESIZE_MAX_PIXELS
    ap = argparse.ArgumentParser(description="Point-OCR marker studio")
    ap.add_argument("--base", type=str, default=None, help="Base VL checkpoint (dir or HF id).")
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=RESIZE_MAX_PIXELS,
        help="Inference resize cap; keep equal to training --max-pixels (default 2048²).",
    )
    ap.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA dir. Combine with --base; mismatch is rejected.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Alias of --base for a bare VL checkpoint.",
    )
    args = ap.parse_args()
    RESIZE_MAX_PIXELS = int(args.max_pixels)
    if args.base or args.model:
        picked = args.base or args.model
        p = Path(picked).expanduser()
        DEFAULT_BASE = str(p.resolve()) if p.exists() else str(picked)
        if not args.adapter and _is_full_vl_checkpoint(Path(DEFAULT_BASE)):
            # A merged checkpoint passed via --base with no --adapter means "no LoRA".
            DEFAULT_LORA = None
    if args.adapter:
        p = Path(args.adapter).expanduser()
        if p.exists() and _is_lora_dir(str(p.resolve() if p.exists() else p)):
            DEFAULT_LORA = p.resolve()
            expected = _lora_base_model(str(DEFAULT_LORA))
            if expected:
                DEFAULT_BASE = expected
        elif p.exists():
            DEFAULT_BASE = str(p.resolve())
            DEFAULT_LORA = None
        else:
            raise SystemExit(f"--adapter not found: {args.adapter}")
    _ensure_probe_image()
    LABEL_INBOX.mkdir(parents=True, exist_ok=True)
    LABEL_OUT.mkdir(parents=True, exist_ok=True)
    LABEL_WORK.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, name="marker-studio-gpu", daemon=True).start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"marker studio  http://{HOST}:{PORT}", flush=True)
    print(f"label tool     http://{HOST}:{PORT}/label", flush=True)
    print(f"default base {DEFAULT_BASE}", flush=True)
    print(f"default lora {DEFAULT_LORA if DEFAULT_LORA and DEFAULT_LORA.is_dir() else '(none)'}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

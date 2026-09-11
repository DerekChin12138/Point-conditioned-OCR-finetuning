# OvisOCR2 Point-Conditioned OCR Finetuning

Self-contained training repo for **point-conditioned semantic-block OCR** on top of [ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) ([technical report](https://arxiv.org/abs/2607.13639)).

**Goal:** given a page/screenshot image with a fixed magenta/white crosshair, the model outputs **only** the minimal semantic block under that point (or empty string on chrome/blank). Product translation UI is out of scope — deliver Stage A weights (+ GGUF).

## Quick map

| Path | Role |
|------|------|
| `src/point_ocr/marker.py` | Frozen `MARKER_SPEC` + draw |
| `src/point_ocr/prompts.py` | Fixed `POINT` / `PAGE` English prompts |
| `data/synth/` | HTML templates + Playwright render arm |
| `data/scripts/` | Build / merge datasets |
| `train/` | Stage A/B QLoRA YAML (CUDA, ~24GB) |
| `eval/` | Held-out build + metrics |
| `export/` | LoRA merge → GGUF (+ mmproj notes) |
| `notebooks/` | **云端逐步点跑入口（推荐）** |
| `docs/FRAMEWORK.md` | 框架详解 + 迁云清单 |

## Environment

- **Package manager:** [uv](https://docs.astral.sh/uv/) (creates `.venv` + `uv.lock`). Do not use bare `python -m venv` / `pip install` for this repo.
- **Train/eval/export:** Windows or Linux + **NVIDIA CUDA**. Do **not** use Apple MLX / macOS-only stacks for training.
- This Mac checkout can still develop data code, markers, and unit tests.

```bash
# Core + dev (editable install of this package)
uv sync --extra dev

# Synth rendering
uv sync --extra dev --extra synth
uv run playwright install chromium

# On CUDA host for train/eval (+ LLaMA-Factory separately for train)
uv sync --extra train
uv sync --extra eval    # vLLM for OvisOCR2 inference/labeling

# Run tools without manual activate
uv run pytest -q
uv run python data/scripts/build_synth_batch.py --help

# Cloud / Jupyter (recommended on rented GPUs)
uv pip install jupyter ipykernel
uv run jupyter lab --ip=0.0.0.0 --port=8888
# then open notebooks/00_cloud_setup.ipynb → … → 06_export.ipynb
```

框架细节与迁云步骤见 [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md)。  
**AutoDL 从安装到训练的命令清单：** [`docs/AUTODL.md`](docs/AUTODL.md)  
**本地样例预览：** 打开 [`data/preview_samples/index.html`](data/preview_samples/index.html)

## Data engine (hard constraints)

1. **One page → many POINT samples:** each block gets **R=2–5** center-biased points; plus negatives on edges/gutters → empty label.
2. **Synth (scale):** HTML with `data-block-id` → Playwright screenshot + DOM bbox → Markdown from **same HTML** (never OCR-as-GT on synth).
3. **Real (15–30%):** layout bboxes → **OvisOCR2 crop OCR** as text GT → mark points on full page. Paddle/MinerU = geometry backup only.
4. **Marker:** always `MARKER_SPEC` (magenta cross + white ring). Train ≡ serve.  
   Block boxes use **text-ink** geometry (`Range.getClientRects`), not full-width element boxes.  
5. **QA:** conservative filters + bucketed spot-checks; drop bad buckets.

**Why Chromium?** Playwright needs a browser engine to paint HTML→pixels and read DOM text boxes. `playwright install chromium` installs that engine — it is unrelated to GPU training.

```bash
# Smoke synth (needs Playwright browser + Linux system libs)
uv run playwright install chromium
uv run playwright install-deps chromium   # if missing: build fails, no point_sharegpt.jsonl
uv run python data/scripts/build_synth_batch.py --out data/processed/synth --noise

# Merge for LLaMA-Factory
python data/scripts/merge_and_filter.py \
  --synth data/processed/synth/point_sharegpt.jsonl \
  --out-dir data/llamafactory
```

Scale targets: ~1k hand-checked pairs first, then 50k–200k.

## Training (Stage A) — Unsloth（推荐）

Base: `ATH-MaaS/OvisOCR2`. Prefer **Unsloth** `FastVisionModel` QLoRA on AutoDL (see [`docs/UNSLOTH.md`](docs/UNSLOTH.md)). Freeze vision by default; 1–3 epochs; early-stop on **over-extraction**.

```bash
uv sync --extra train
uv pip install unsloth
uv run python train/unsloth_stage_a.py --smoke-load-only   # first time
bash train/run_unsloth.sh
# → checkpoints/<YYYYMMDD_HHMMSS>_stage_a/{run_config.json,tb/,checkpoint-*,adapter_final/,metrics/}
```

LLaMA-Factory YAML remains as fallback: `train/stage_a_point_qlora.yaml`.

## Eval (build set before trusting train)

Metrics: `block_hit_rate`, `mean_normalized_edit_distance`, `over_extraction_rate`, `empty_on_chrome_rate` (also written to each run’s `metrics/final_report.json`).

```bash
python eval/build_heldout.py
python eval/run_eval.py --backend vllm --model /path/to/finetuned_or_base
python eval/baselines/zero_shot_ovis.py   # stock OvisOCR2 + POINT prompt
```

**Go:** on identical full-screen + crosshair cases, finetuned mainly returns the pointed block; clear win vs zero-shot dump.

**No-Go:** systematic full-page dump → fix marker alignment, label noise, negatives — not “bigger base”.

## Export

See [`export/README.md`](export/README.md): merge LoRA → `Q4_K_M` / `Q5_K_M` GGUF (+ mmproj if runtime needs it).

## Stage B (optional)

Only after Stage A Go. Mix PAGE + POINT with different user prompts in one weight (`train/stage_b_dual_qlora.yaml`). If PAGE collapses or POINT regresses, keep independent POINT weights.

## Docs

- [`docs/TASK_BRIEF.md`](docs/TASK_BRIEF.md) — full task charter  
- [`docs/MARKER_SPEC.md`](docs/MARKER_SPEC.md) — visual protocol  
- [`docs/PROMPTS.md`](docs/PROMPTS.md) — prompt contract  
- [`docs/STAGES.md`](docs/STAGES.md) — Stage A/B checklist  

## Tests (offline)

```bash
uv sync --extra dev
uv run pytest -q
```

## License

Apache-2.0 for this repo’s code. Base model / data you collect have their own terms — keep private screenshots private.

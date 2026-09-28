#!/usr/bin/env bash
# Launch the marker studio on the current Q2 (point-OCR + EN→ZH) model.
#
#   bash eval/marker_studio/run_q2.sh
#
# Current deliverable: checkpoints/q2_2b_merged = Qwen3.5-2B + Q1 SFT + Q2 SFT,
# fully merged (NO LoRA). Prompt preset: `ocr_mt_v1`（点OCR + 英译中）.
#
# Pages:
#   probe  http://127.0.0.1:7865        place a crosshair, tweak size/decode/prompt
#   label  http://127.0.0.1:7865/label  draw boxes on real screenshots, write block text
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Full merged model (no adapter). Override BASE/ADAPTER for other checkpoints.
BASE="${BASE:-checkpoints/q2_2b_merged}"
ADAPTER="${ADAPTER:-}"
# Inference resolution: keep equal to training --max-pixels (default 2048²).
MAX_PIXELS="${MAX_PIXELS:-4194304}"

if [[ ! -f "$BASE/config.json" ]]; then
  echo "base not found (need config.json): $BASE" >&2
  echo "available merged checkpoints:" >&2
  for d in checkpoints/*/; do
    [[ -f "$d/config.json" && ! -f "$d/adapter_config.json" ]] && echo "  $d" >&2
  done
  exit 1
fi

echo "studio base    = $BASE"
if [[ -n "$ADAPTER" ]]; then echo "studio adapter = $ADAPTER"; else echo "studio adapter = (none — merged model)"; fi
echo "prompt preset  = ocr_mt_v1（点OCR + 英译中）; a2_v3 可单独验证 Q1 跟点"
echo "probe          = http://127.0.0.1:7865"
echo "label          = http://127.0.0.1:7865/label"

if [[ -n "$ADAPTER" ]]; then
  exec uv run python eval/marker_studio/server.py --base "$BASE" --adapter "$ADAPTER" --max-pixels "$MAX_PIXELS" "$@"
fi
exec uv run python eval/marker_studio/server.py --base "$BASE" --max-pixels "$MAX_PIXELS" "$@"

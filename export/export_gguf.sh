#!/usr/bin/env bash
# Export merged HF weights to llama.cpp GGUF (+ mmproj if required by the VL stack).
#
# Prereqs on export host:
#   - clone llama.cpp and build
#   - python convert script matching Qwen3.5-VL / OvisOCR2 architecture
#
# Usage:
#   export LLAMA_CPP_ROOT=/path/to/llama.cpp
#   ./export/export_gguf.sh /path/to/merged_hf exports/OvisOCR2-Point-Q4_K_M.gguf Q4_K_M

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_DIR="${1:?merged HF model dir}"
OUT_GGUF="${2:?output .gguf path}"
QUANT="${3:-Q4_K_M}"

if [[ -z "${LLAMA_CPP_ROOT:-}" ]]; then
  echo "Set LLAMA_CPP_ROOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_GGUF")"
FP16_GGUF="${OUT_GGUF%.gguf}.f16.gguf"

# Convert script name varies by llama.cpp version / model family — try common entrypoints
if [[ -f "$LLAMA_CPP_ROOT/convert_hf_to_gguf.py" ]]; then
  python "$LLAMA_CPP_ROOT/convert_hf_to_gguf.py" "$HF_DIR" --outfile "$FP16_GGUF"
elif [[ -f "$LLAMA_CPP_ROOT/convert-hf-to-gguf.py" ]]; then
  python "$LLAMA_CPP_ROOT/convert-hf-to-gguf.py" "$HF_DIR" --outfile "$FP16_GGUF"
else
  echo "Could not find convert_hf_to_gguf.py under $LLAMA_CPP_ROOT" >&2
  exit 1
fi

QUANTIZE="$LLAMA_CPP_ROOT/llama-quantize"
if [[ ! -x "$QUANTIZE" ]]; then
  QUANTIZE="$LLAMA_CPP_ROOT/build/bin/llama-quantize"
fi
"$QUANTIZE" "$FP16_GGUF" "$OUT_GGUF" "$QUANT"

echo "Wrote $OUT_GGUF"
echo "If your runtime needs a separate mmproj, export it with the matching"
echo "llama.cpp multimodal projector conversion for this vision tower and place"
echo "beside the GGUF (see export/README.md)."

#!/usr/bin/env bash
# Launch Stage A on a CUDA host (Linux/Windows Git Bash / WSL).
# Usage:
#   export LLAMA_FACTORY_ROOT=/path/to/LLaMA-Factory
#   ./train/run_train.sh stage_a
#   ./train/run_train.sh stage_b

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${1:-stage_a}"

if [[ -z "${LLAMA_FACTORY_ROOT:-}" ]]; then
  echo "Set LLAMA_FACTORY_ROOT to your LLaMA-Factory clone." >&2
  exit 1
fi

case "$STAGE" in
  stage_a) CFG="$ROOT/train/stage_a_point_qlora.yaml" ;;
  stage_b) CFG="$ROOT/train/stage_b_dual_qlora.yaml" ;;
  *) echo "Unknown stage: $STAGE (stage_a|stage_b)" >&2; exit 1 ;;
esac

# Paths inside YAML are relative to train/; run from train/
cd "$ROOT/train"
python "$LLAMA_FACTORY_ROOT/src/train.py" "$CFG" \
  || llamafactory-cli train "$CFG"

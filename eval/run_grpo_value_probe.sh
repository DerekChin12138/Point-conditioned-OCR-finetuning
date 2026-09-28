#!/usr/bin/env bash
# G=8 GRPO value probe on q1_withreal hard scenes. Linux + NVIDIA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p checkpoints/grpo_value_probe
exec uv run python eval/run_grpo_value_probe.py "$@"

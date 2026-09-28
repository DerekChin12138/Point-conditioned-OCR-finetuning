#!/usr/bin/env bash
# One shot: A1 then A2 from A1 adapter_final.
#   bash train/run_two_stage.sh
#   bash train/run_two_stage.sh --wait-data
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PY:-python3}"
fi
exec "$PY" -u "$ROOT/train/run_two_stage.py" "$@"

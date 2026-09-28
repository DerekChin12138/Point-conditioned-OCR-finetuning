#!/usr/bin/env bash
# Sequential Q1 pool build from Qwen Instruct recipe. Does not touch data/pools or data/pools_b.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
LOGDIR="$ROOT/data/pools_q/logs"
mkdir -p "$LOGDIR"
W="${WORKERS:-12}"

run_pool() {
  local id="$1" target="$2" seed="$3"
  echo "[seq] start $id target=$target workers=$W $(date -Iseconds)" | tee -a "$LOGDIR/sequential.log"
  uv run python data/scripts/build_pool.py \
    --pool-id "$id" \
    --out-root data/pools_q \
    --target "$target" \
    --chrome \
    --wipe \
    --workers "$W" \
    --seed "$seed" \
    --prompt-key a2_v3 \
    | tee -a "$LOGDIR/${id}.log"
  echo "[seq] done $id $(date -Iseconds)" | tee -a "$LOGDIR/sequential.log"
}

run_pool semantic_group 2800 41
run_pool empty_special 3500 46
run_pool empty_clear 5000 43
run_pool multi_frag 2800 45
run_pool core_inner 18000 42

echo "[seq] all pools done $(date -Iseconds)" | tee -a "$LOGDIR/sequential.log"

#!/usr/bin/env bash
# Sequential B1 pool build. Does not touch data/pools (A1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
LOGDIR="$ROOT/data/pools_b/logs"
mkdir -p "$LOGDIR"
W="${WORKERS:-12}"

run_pool() {
  local id="$1" target="$2" seed="$3"
  echo "[seq] start $id target=$target workers=$W $(date -Iseconds)" | tee -a "$LOGDIR/sequential.log"
  uv run python data/scripts/build_pool.py \
    --pool-id "$id" \
    --out-root data/pools_b \
    --target "$target" \
    --chrome \
    --wipe \
    --workers "$W" \
    --seed "$seed" \
    | tee -a "$LOGDIR/${id}.log"
  echo "[seq] done $id $(date -Iseconds)" | tee -a "$LOGDIR/sequential.log"
}

# semantic_group already complete at 2800
run_pool empty_boundary 3000 44
run_pool multi_frag 2800 45
run_pool empty_clear 7000 43
run_pool core_inner 18000 42

echo "[seq] all pools done $(date -Iseconds)" | tee -a "$LOGDIR/sequential.log"

#!/usr/bin/env bash
# Sequential Q2 OCR-MT pool build. Does not touch data/pools_q.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
LOGDIR="$ROOT/data/pools_q2/logs"
mkdir -p "$LOGDIR"
W="${WORKERS:-12}"
POOL="$ROOT/data/synth/content_pools/q2_opus_en_zh.json"
if [[ ! -f "$POOL" ]]; then
  echo "missing $POOL — run data/scripts/build_opus_bilingual_pool.py first" >&2
  exit 1
fi

run_pool() {
  local id="$1" target="$2" seed="$3"
  echo "[seq] start $id target=$target workers=$W $(date -Iseconds)" | tee -a "$LOGDIR/sequential.log"
  uv run python data/scripts/build_pool.py \
    --pool-id "$id" \
    --out-root data/pools_q2 \
    --content-pool "$POOL" \
    --target "$target" \
    --chrome \
    --wipe \
    --workers "$W" \
    --seed "$seed" \
    --prompt-key ocr_mt_v1 \
    --fill-static \
    --pairs-per-bucket 12 \
    | tee -a "$LOGDIR/${id}.log"
  echo "[seq] done $id $(date -Iseconds)" | tee -a "$LOGDIR/sequential.log"
}

# Headroom over compose quotas (30k / 7.5k / 5k / 7.5k / 3.75k).
run_pool semantic_group 4200 41
run_pool empty_special 5500 46
run_pool empty_clear 8000 43
run_pool multi_frag 8000 45
run_pool core_inner 32000 42

echo "[seq] all pools done $(date -Iseconds)" | tee -a "$LOGDIR/sequential.log"

#!/usr/bin/env bash
# Append chrome-force samples so Q2 compose can keep 40% chrome.
# Does not wipe data/pools_q2/{pool_id}.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
W="${WORKERS:-12}"
POOL="$ROOT/data/synth/content_pools/q2_opus_en_zh.json"
TOP="$ROOT/data/pools_q2/_chrome_topup"

topup() {
  local id="$1" extra="$2" seed="$3"
  echo "[topup] start $id extra=$extra seed=$seed workers=$W $(date -Iseconds)"
  uv run python data/scripts/build_pool.py \
    --pool-id "$id" \
    --out-root "$TOP" \
    --content-pool "$POOL" \
    --target "$extra" \
    --chrome \
    --chrome-force \
    --wipe \
    --workers "$W" \
    --seed "$seed" \
    --prompt-key ocr_mt_v1 \
    --mix-marker-scales \
    --fill-static \
    --pairs-per-bucket 12
  local src="$TOP/$id/point_sharegpt.jsonl"
  local dst="$ROOT/data/pools_q2/$id/point_sharegpt.jsonl"
  local meta="$ROOT/data/pools_q2/$id/pool_meta.json"
  [[ -f "$src" ]] || { echo "missing $src" >&2; exit 1; }
  cat "$src" >> "$dst"
  python3 - "$dst" "$meta" "$extra" <<'PY'
import json, sys
from pathlib import Path
jsonl, meta_path, extra = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
n = sum(1 for _ in jsonl.open())
meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
meta["n_written"] = n
meta["chrome_topup"] = int(meta.get("chrome_topup") or 0) + extra
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
print(f"[topup] merged {extra} → {jsonl} n_written={n}", flush=True)
PY
  echo "[topup] done $id $(date -Iseconds)"
}

# Shortages vs 40% chrome compose quotas, plus buffer.
topup semantic_group 700 1041
topup empty_special 700 1046
topup multi_frag 400 1045
topup core_inner 1600 1042

echo "[topup] all done $(date -Iseconds)"

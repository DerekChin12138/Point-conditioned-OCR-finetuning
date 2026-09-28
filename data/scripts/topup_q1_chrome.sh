#!/usr/bin/env bash
# Append chrome-force samples so the Q1 compose can keep 40% chrome.
# Does not wipe data/pools_q/<pool_id>.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
W="${WORKERS:-12}"
TOP="$ROOT/data/pools_q/_chrome_topup"

topup() {
  local id="$1" extra="$2" seed="$3"
  echo "[topup] start $id extra=$extra seed=$seed workers=$W $(date -Iseconds)"
  uv run python data/scripts/build_pool.py \
    --pool-id "$id" \
    --out-root "$TOP" \
    --target "$extra" \
    --chrome \
    --chrome-force \
    --wipe \
    --workers "$W" \
    --seed "$seed" \
    --prompt-key a2_v3
  local src="$TOP/$id/point_sharegpt.jsonl"
  local dst="$ROOT/data/pools_q/$id/point_sharegpt.jsonl"
  local meta="$ROOT/data/pools_q/$id/pool_meta.json"
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

# semantic_group: 569 chrome vs 750 needed at the 40% compose quota → +300 (buffer).
topup semantic_group 300 2401

echo "[topup] all done $(date -Iseconds)"

#!/usr/bin/env bash
# =============================================================================
# Package this workspace for cloud training (弱网 / AutoDL / 任意 CUDA 主机).
#
#   bash scripts/pack_for_cloud.sh --dry-run          # 只盘点，不打包
#   bash scripts/pack_for_cloud.sh                    # 产出 dist/cloud_pack/{code_and_splits.tar.gz,images.tar,SHA256SUMS}
#   bash scripts/pack_for_cloud.sh --code-only        # 只打包代码+split（<300MB，先上云试跑）
#   bash scripts/pack_for_cloud.sh --with-synth       # 附带 data/synth（重渲页面时需要）
#   bash scripts/pack_for_cloud.sh --with-grpo        # 附带 GRPO split 引用的图
#   bash scripts/pack_for_cloud.sh --with-checkpoints # 附带 checkpoints/*_merged（续训用）
#
# 设计：
#   * 代码 + split + recipes  → code_and_splits.tar.gz（可 gzip，几十 MB）
#   * split 引用到的 marked 图 → images.tar（JPEG 不可压，不 gzip；~20 GiB）
#   * 模型（models/）**不打包**：云端用 hf-mirror 重下，省 15 GiB 上行
#   * .env 不入包（含 API key）；云端按需自建
#   * 绝对路径问题：split 里存的是本机绝对路径 → 云端用 scripts/relocate_paths.py 重定位
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="dist/cloud_pack"
CODE_ONLY=0
WITH_SYNTH=0
WITH_GRPO=0
WITH_CHECKPOINTS=0
DRY_RUN=0
STAMP="$(date +%Y%m%d)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --code-only) CODE_ONLY=1; shift ;;
    --with-synth) WITH_SYNTH=1; shift ;;
    --with-grpo) WITH_GRPO=1; shift ;;
    --with-checkpoints) WITH_CHECKPOINTS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
CODE_LIST="$OUT/code_files.txt"
IMG_LIST="$OUT/images.txt"
: > "$CODE_LIST"
: > "$IMG_LIST"

# --- 1) 代码 / 配置 / split / recipes -----------------------------------------
CODE_PATHS=(
  src train eval export scripts tests
  data/scripts data/recipes
  docs notebooks
  README.md LICENSE pyproject.toml uv.lock .python-version .env.example .gitignore
)
CODE_EXTS=(py sh md yaml yml toml json html jinja ipynb txt cfg)
find_code() {
  local p="$1"
  if [[ -f "$p" ]]; then
    echo "$p"
  elif [[ -d "$p" ]]; then
    local args=()
    for e in "${CODE_EXTS[@]}"; do args+=(-o -name "*.$e"); done
    find "$p" -type f \( "${args[@]:1}" \) \
      -not -path '*/__pycache__/*' -not -path '*/.pytest_cache/*' \
      -not -path '*/.git/*' -not -path '*/node_modules/*'
  fi
}
for p in "${CODE_PATHS[@]}"; do find_code "$p"; done >> "$CODE_LIST"

# splits (jsonl) — 训练/评测的清单
for d in data/splits_stage_q1_withreal data/splits_stage_q2_ocr_mt_v2; do
  [[ -d "$d" ]] && find "$d" -type f -name '*.jsonl' >> "$CODE_LIST"
done
if [[ "$WITH_GRPO" -eq 1 ]]; then
  for d in data/splits_grpo_* ; do [[ -d "$d" ]] && find "$d" -type f -name '*.jsonl' >> "$CODE_LIST"; done
fi
if [[ "$WITH_SYNTH" -eq 1 ]]; then
  find data/synth -type f \
    \( -name '*.html' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.png' -o -name '*.jpg' -o -name '*.py' \) \
    -not -path '*/__pycache__/*' >> "$CODE_LIST"
fi
if [[ "$WITH_CHECKPOINTS" -eq 1 ]]; then
  for d in checkpoints/*_merged checkpoints/*/adapter_final ; do
    [[ -d "$d" ]] && find "$d" -type f \
      \( -name '*.safetensors' -o -name '*.json' -o -name '*.jinja' -o -name '*.txt' -o -name '*.model' \) >> "$CODE_LIST"
  done
fi
sort -u -o "$CODE_LIST" "$CODE_LIST"

# --- 2) split 引用到的图片 ----------------------------------------------------
SPLIT_DIRS=("data/splits_stage_q1_withreal" "data/splits_stage_q2_ocr_mt_v2")
[[ "$WITH_GRPO" -eq 1 ]] && for d in data/splits_grpo_*; do [[ -d "$d" ]] && SPLIT_DIRS+=("$d"); done
if [[ "$CODE_ONLY" -eq 0 ]]; then
  uv run python - "$OUT/images.txt" "${SPLIT_DIRS[@]}" <<'PY'
import json, os, sys
out_path = sys.argv[1]
dirs = sys.argv[2:]
seen, missing = {}, []
for d in dirs:
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jsonl"):
            continue
        for line in open(os.path.join(d, fn), encoding="utf-8"):
            if not line.strip():
                continue
            imgs = (json.loads(line).get("images") or [])
            if not imgs:
                continue
            p = imgs[0]
            seen[p] = None
with open(out_path, "w", encoding="utf-8") as f:
    for p in sorted(seen):
        if os.path.exists(p):
            f.write(os.path.relpath(p, os.getcwd()) + "\n")
        else:
            missing.append(p)
print(f"[images] unique referenced={len(seen)} present={len(seen)-len(missing)} missing={len(missing)}", flush=True)
for p in missing[:10]:
    print(f"  [missing] {p}", file=sys.stderr)
PY
fi

# --- 3) manifest --------------------------------------------------------------
total_human() {
  python3 - "$1" <<'PY'
import os, sys
n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    p = line.strip()
    if p and os.path.exists(p):
        n += os.path.getsize(p)
u = ["B", "KiB", "MiB", "GiB", "TiB"]
i = 0
f = float(n)
while f >= 1024 and i < 4:
    f /= 1024
    i += 1
print(f"{f:.1f} {u[i]}")
PY
}
N_CODE=$(wc -l < "$CODE_LIST")
N_IMG=$(wc -l < "$IMG_LIST")
SZ_CODE=$(total_human "$CODE_LIST")
SZ_IMG=$([[ "$N_IMG" -gt 0 ]] && total_human "$IMG_LIST" || echo "0")
echo "===================== cloud pack manifest ====================="
echo "code/config/split files : $N_CODE  ($SZ_CODE)"
echo "referenced marked images: $N_IMG  ($SZ_IMG)"
echo "out dir                 : $OUT"
echo "==============================================================="

cat > "$OUT/README_PACK.txt" <<EOF
point-ocr workspace pack  ($(date '+%F %T'))
code_and_splits.tar.gz : code + configs + splits${WITH_SYNTH:+ + synth templates}
images.tar             : marked images referenced by the splits
models/                : NOT included — download on the cloud (hf-mirror)
.env                   : NOT included — recreate on the cloud
after extracting:  uv run python scripts/relocate_paths.py --old-root $ROOT --apply
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] lists written: $CODE_LIST , $IMG_LIST (nothing tarred)"
  exit 0
fi

# --- 4) tarballs --------------------------------------------------------------
CODE_TAR="$OUT/code_and_splits_$STAMP.tar.gz"
IMG_TAR="$OUT/images_$STAMP.tar"
echo "[pack] $CODE_TAR"
tar -czf "$CODE_TAR" -C "$ROOT" -T "$CODE_LIST"
if [[ "$N_IMG" -gt 0 ]]; then
  echo "[pack] $IMG_TAR  (this is the big one, JPEGs are already compressed)"
  tar -cf "$IMG_TAR" -C "$ROOT" -T "$IMG_LIST"
fi
( cd "$OUT" && sha256sum ./*.tar ./*.tar.gz 2>/dev/null > SHA256SUMS )
echo "[pack] done:"
ls -lh "$OUT"
echo
echo "next: rsync/scp 到云端，或 AutoDL 控制台上传 $OUT/"
echo "  rsync -avP --partial -e 'ssh -p <PORT>' $OUT/ root@<HOST>:/root/autodl-tmp/ptocr_pack/"

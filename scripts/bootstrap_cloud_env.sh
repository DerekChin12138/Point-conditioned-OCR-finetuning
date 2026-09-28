#!/usr/bin/env bash
# =============================================================================
# 云端（AutoDL/任意 CUDA 主机）一键引导：装 uv → 配镜像 → 装依赖 → 验证。
#
#   cd <项目根>
#   bash scripts/bootstrap_cloud_env.sh
#
# 幂等：已装好的步骤会跳过。国内网络优先走 pip 镜像装 uv，失败再退回官方脚本。
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "[bootstrap] project root = $ROOT"

# --- 1) 镜像（提供 PIP_INDEX_URL / UV_INDEX_URL / HF_ENDPOINT / GITHUB_PROXY）---
# shellcheck disable=SC1091
source scripts/setup_autodl_mirrors.sh

# --- 2) uv -------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
install_uv_pip() {
  local py
  for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
      echo "[bootstrap] trying: $py -m pip install -U uv"
      if "$py" -m pip install -U uv -i "${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"; then
        return 0
      fi
    fi
  done
  return 1
}
if command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] uv already installed"
elif install_uv_pip; then
  :
else
  echo "[bootstrap] pip route failed; trying astral installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh || \
    curl -LsSf "${GITHUB_PROXY}/https://astral.sh/uv/install.sh" | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "[bootstrap] uv 仍然不可用，请手动安装" >&2; exit 1; }
uv --version

# uv 需要 Python 3.12（见 .python-version）；若本机没有，uv 会去 GitHub 下，
# 国内可能很慢 —— 走 ghproxy 镜像。
export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-${GITHUB_PROXY}/https://github.com/astral-sh/python-build-standalone/releases/download}"

# --- 3) 依赖 -----------------------------------------------------------------
uv sync --extra dev --extra train --extra synth
uv pip install unsloth

# --- 4) 验证 -----------------------------------------------------------------
uv run python - <<'PY'
import torch, transformers, sys
print("[bootstrap] python", sys.version.split()[0])
print("[bootstrap] torch", torch.__version__, "cuda:", torch.cuda.is_available())
print("[bootstrap] transformers", transformers.__version__)
PY
if uv run hf --help >/dev/null 2>&1; then
  echo "[bootstrap] hf CLI OK  (huggingface_hub: $(uv run python -c 'import huggingface_hub as h;print(h.__version__)'))"
else
  echo "[bootstrap][warn] hf CLI 不可用，可改用 'uv run huggingface-cli download ...'" >&2
fi

cat <<'EOF'

[bootstrap] 完成。接下来（在项目根执行）：
  uv run pytest -q
  uv run hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
  uv run hf download Qwen/Qwen3.5-2B   --local-dir models/Qwen3.5-2B
  uv run hf download Qwen/Qwen3.5-4B   --local-dir models/Qwen3.5-4B
  uv run python scripts/relocate_paths.py \
    --old-root /home/derek_qxc/workspace/ocr-finetuning/Point-conditioned-OCR-finetuning --apply
  bash train/run_08b_pipeline.sh smoke
EOF

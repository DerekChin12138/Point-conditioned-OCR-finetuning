#!/usr/bin/env bash
# AutoDL / 国内弱网：一键配置常用下载镜像。
# 用法（每个新终端先执行一次）：
#   source scripts/setup_autodl_mirrors.sh
#
# 不要用 `bash scripts/...`（那样 export 进不了当前 shell）。

# --- PyPI / uv ---
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# 兼容部分仍走 pip 的工具
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com pypi.tuna.tsinghua.edu.cn files.pythonhosted.org}"

# --- Hugging Face（模型/数据集）---
# 官方: https://huggingface.co  → 镜像:
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 可选：缓存目录放到数据盘（按需改路径）
# export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
# export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

# --- Playwright Chromium ---
export PLAYWRIGHT_DOWNLOAD_HOST="${PLAYWRIGHT_DOWNLOAD_HOST:-https://npmmirror.com/mirrors/playwright}"

# --- 可选：GitHub 资源（curl/wget 自己拼；git clone 见下方提示）---
export GITHUB_PROXY="${GITHUB_PROXY:-https://ghproxy.net}"

echo "[mirrors] UV_INDEX_URL=$UV_INDEX_URL"
echo "[mirrors] HF_ENDPOINT=$HF_ENDPOINT"
echo "[mirrors] PLAYWRIGHT_DOWNLOAD_HOST=$PLAYWRIGHT_DOWNLOAD_HOST"
echo "[mirrors] GITHUB_PROXY=$GITHUB_PROXY"
echo "[mirrors] OK — 继续 uv sync / playwright install / 下载模型即可"
echo "[mirrors] git clone 示例:"
echo "  git -c http.version=HTTP/1.1 clone --depth 1 \\"
echo "    \${GITHUB_PROXY}/https://github.com/DerekChin12138/Point-conditioned-OCR-finetuning.git"

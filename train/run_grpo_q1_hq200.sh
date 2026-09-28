#!/usr/bin/env bash
# 兼容旧名：等价于 `KIND=q1 bash train/run_grpo_hq.sh <stage>`。
# 通用脚本（支持 Q1/Q2 × 0.8B/2B/4B）见 train/run_grpo_hq.sh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env KIND=q1 bash "$ROOT/train/run_grpo_hq.sh" "$@"
